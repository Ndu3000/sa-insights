"""Knowledge graph over the SA Insights semantic model.

Why a graph at all
------------------
Thirty-three series is small, but the *questions* are relational: "how has the
economy improved" is not one series, it is a set of them plus the definitions
that connect them (a real policy rate is the repo rate relative to inflation).
Vector search alone retrieves text that looks similar; it cannot tell you that
``policy_rate`` and ``cpi_yoy`` combine into ``real_policy_rate``, nor that
``lending_rate`` is downstream of ``policy_rate``.

The graph is built from ``sa_insights.semantic`` — the same model that is
published as the Apache Ossie artefact — so there is exactly one description of
what the data means, and the chatbot cannot drift from the exports.

Why not Neo4j
-------------
33 series and ~175 metrics fit in memory with room to spare, so a graph server
would add an operational dependency to traverse a few hundred nodes. The
structure is real; the infrastructure is not warranted. ``Graph.neighbours``
does the traversal that matters (bounded multi-hop, typed edges).

Node and edge types
-------------------
    series   --measured_by-->  metric
    metric   --derived_from->  series     (metric needs these inputs)
    series   --in_category-->  category
    metric   --in_category-->  category
    series   --compared_to-->  series     (same category + same frequency)
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

# Edge weights used when ranking a traversal. A metric that *derives* from a
# series is a strong semantic link (it explains the series); co-membership in a
# category is weak (it only says they are adjacent in the catalogue).
WEIGHT = {
    "measured_by": 1.0,
    "derived_from": 0.9,
    "in_category": 0.4,
    "compared_to": 0.3,
    "about_topic": 0.8,
}

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in into is it its of on or
    that the their to was were what when where which who why will with""".split()
)

# Themes: the bridge between how people ask and how the data is labelled.
#
# "How has the economy improved" matches no series label — not one contains the
# word "economy" — so lexical scoring alone returns nothing for the most natural
# possible question. These nodes give everyday phrasing somewhere to land, and
# the edges carry the domain judgement about which series belong together when
# someone asks a question that broad.
THEMES: dict[str, dict] = {
    "economy": {
        "label": "the broader economy",
        "description": "Overall economic performance: output, growth and demand.",
        "synonyms": [
            "economy",
            "economic",
            "macro",
            "macroeconomic",
            "growth",
            "doing",
            "performing",
            "overall",
            "recovery",
            "recession",
            "boom",
        ],
        "members": ["gdp_growth", "gdp_real", "leading_indicator", "retail_sales", "manufacturing"],
    },
    "cost_of_living": {
        "label": "the cost of living",
        "description": "Inflation and prices facing households.",
        "synonyms": [
            "inflation",
            "prices",
            "cost of living",
            "expensive",
            "cpi",
            "cost",
            "purchasing power",
            "price",
        ],
        "members": ["cpi_yoy", "ppi_yoy", "weo_inflation", "inflation_annual"],
    },
    "job_market": {
        "label": "the labour market",
        "description": "Employment, unemployment and participation.",
        "synonyms": [
            "jobs",
            "job",
            "unemployment",
            "employment",
            "labour",
            "labor",
            "work",
            "hiring",
            "workforce",
            "participation",
        ],
        "members": ["unemployment", "lf_participation", "gdp_growth"],
    },
    "interest_rates": {
        "label": "interest rates",
        "description": "Policy rate, lending rates and the cost of credit.",
        "synonyms": [
            "rates",
            "rate",
            "interest",
            "repo",
            "borrowing",
            "credit",
            "monetary",
            "tight",
            "loose",
            "lending",
        ],
        "members": ["policy_rate", "lending_rate", "bond_yield"],
    },
    "currency": {
        "label": "the rand",
        "description": "Exchange rate and external competitiveness.",
        "synonyms": ["rand", "currency", "exchange", "fx", "dollar", "zar", "weaker", "stronger"],
        "members": ["usd_zar", "reer"],
    },
    "external": {
        "label": "external position",
        "description": "Trade, current account and foreign balance.",
        "synonyms": ["trade", "exports", "imports", "current account", "external", "balance"],
        "members": ["current_account", "exports_usd", "imports_usd"],
    },
    "fiscal": {
        "label": "government finances",
        "description": "Fiscal balance, debt and the public purse.",
        "synonyms": ["debt", "fiscal", "budget", "deficit", "government", "treasury", "borrowing"],
        "members": ["fiscal_balance", "govt_debt", "bond_yield"],
    },
}


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed.

    Deliberately crude: this feeds lexical scoring, where stemming or
    lemmatisation would add a dependency and, at this corpus size, change
    almost nothing. Words are also matched against their singular form so that
    "forecasts" and "forecast" are the same key.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for word in words:
        if word in _STOPWORDS or len(word) < 2:
            continue
        out.append(word)
        if word.endswith("s") and len(word) > 3:
            out.append(word[:-1])
    return out


@dataclass
class Node:
    """One graph vertex. ``label`` is what a human reads, ``key`` is identity."""

    key: str
    kind: str  # series | metric | category
    label: str
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def search_text(self) -> str:
        """Everything that should count when matching a user's phrasing."""
        parts = [self.key, self.label, self.description, *self.synonyms]
        return " ".join(p for p in parts if p)


@dataclass
class Graph:
    """A small typed property graph with bounded traversal."""

    nodes: dict[str, Node] = field(default_factory=dict)
    # adjacency: key -> list of (neighbour_key, edge_type, weight_override)
    _edges: dict[str, list[tuple[str, str, float | None]]] = field(default_factory=dict)
    # inverted index for lexical lookup, key -> term -> tf
    _terms: dict[str, dict[str, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def add_node(self, node: Node) -> None:
        self.nodes[node.key] = node
        counts: dict[str, int] = {}
        for term in _tokens(node.search_text()):
            counts[term] = counts.get(term, 0) + 1
        self._terms[node.key] = counts

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float | None = None,
    ) -> None:
        """Add a directed edge. Unknown endpoints are ignored, not an error.

        The semantic model is built from whatever series actually loaded, so a
        metric can legitimately reference a series that failed to fetch. Dropping
        the edge keeps the graph internally consistent instead of raising on a
        partial refresh — a chatbot that works with 32 of 33 series is far more
        useful than one that refuses to start.

        ``weight`` overrides the per-type default, which is how a theme's most
        representative series can outrank the rest of its members.
        """
        if source not in self.nodes or target not in self.nodes or source == target:
            return
        self._edges.setdefault(source, []).append((target, edge_type, weight))
        self._edges.setdefault(target, []).append((source, edge_type, weight))

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def neighbours(self, key: str, edge_types: Iterable[str] | None = None) -> list[str]:
        allowed = set(edge_types) if edge_types else None
        return [
            other
            for other, kind, _ in self._edges.get(key, [])
            if allowed is None or kind in allowed
        ]

    def edge_type(self, source: str, target: str) -> str | None:
        for other, kind, _ in self._edges.get(source, []):
            if other == target:
                return kind
        return None

    def of_kind(self, kind: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == kind]

    def expand(
        self,
        seeds: Iterable[str],
        hops: int = 2,
        min_weight: float = 0.3,
    ) -> dict[str, float]:
        """Weighted multi-hop expansion from seed nodes.

        Returns node key -> accumulated score, where each hop multiplies by the
        edge weight and decays. Decay is what stops a two-hop walk from
        returning most of the graph: without it, ``policy_rate`` reaches nearly
        everything within three hops and the ranking stops discriminating.
        """
        scores: dict[str, float] = {}
        frontier: deque[tuple[str, int, float]] = deque()

        for seed in seeds:
            if seed in self.nodes:
                scores[seed] = 1.0
                frontier.append((seed, 0, 1.0))

        while frontier:
            key, depth, score = frontier.popleft()
            if depth >= hops:
                continue
            for other, kind, override in self._edges.get(key, []):
                weight = override if override is not None else WEIGHT.get(kind, 0.2)
                if weight < min_weight:
                    continue
                # 0.7^depth: hop one keeps most of its weight, hop two is a hint.
                candidate = score * weight * (0.7**depth)
                if candidate <= scores.get(other, 0.0):
                    continue
                scores[other] = candidate
                frontier.append((other, depth + 1, candidate))

        return scores

    # ------------------------------------------------------------------ #
    # lexical retrieval
    # ------------------------------------------------------------------ #
    def score(self, query: str, kinds: Iterable[str] | None = None) -> dict[str, float]:
        """BM25-ish lexical relevance of every node to a query.

        IDF is computed over nodes, so a term appearing in every description
        ("inflation") contributes almost nothing while a rare one ("repo")
        dominates. This is what makes short, specific questions work without an
        embedding model — and it costs nothing per query.
        """
        wanted = set(kinds) if kinds else None
        terms = _tokens(query)
        if not terms:
            return {}

        total = max(len(self.nodes), 1)
        df: dict[str, int] = {}
        for counts in self._terms.values():
            for term in terms:
                if term in counts:
                    df[term] = df.get(term, 0) + 1

        results: dict[str, float] = {}
        for key, counts in self._terms.items():
            node = self.nodes[key]
            if wanted is not None and node.kind not in wanted:
                continue
            # Exact key or label match is a very strong signal: someone typing
            # "cpi" wants the CPI series, not the corpus's most similar prose.
            bonus = 0.0
            if query.strip().lower() in (node.key, node.label.lower()):
                bonus = 6.0

            score = bonus
            for term in terms:
                tf = counts.get(term)
                if not tf:
                    continue
                n = df.get(term, 0) or 1
                idf = 1.0 + (total / n)
                score += (tf / (tf + 1.2)) * idf
            if score > 0:
                results[key] = score
        return results

    def __len__(self) -> int:
        return len(self.nodes)


def build_graph(series_rows: Iterable[dict], model: dict | None = None) -> Graph:
    """Build the graph from warehouse rows and the Ossie semantic model.

    ``series_rows`` come from ``store.indicator_metadata()`` — one dict per
    series with series_key/label/category/unit/frequency/description. ``model``
    is the dict from ``semantic.build_model()``; when it is absent the graph
    still builds, just without the derived metric layer.
    """
    graph = Graph()

    for row in series_rows:
        # Tolerate both the warehouse column names and plain dicts, so callers
        # can pass a DataFrame's .to_dict("records") without renaming first.
        key = row.get("series_key") or row.get("key")
        if not key:
            continue
        graph.add_node(
            Node(
                key=key,
                kind="series",
                label=row.get("label") or key,
                description=row.get("description") or "",
                synonyms=[row["short_label"]] if row.get("short_label") else [],
                meta={
                    "category": row.get("category"),
                    "unit": row.get("unit"),
                    "frequency": row.get("frequency"),
                    "higher_is_better": row.get("higher_is_better"),
                },
            )
        )

    # Categories, and series-to-series "adjacent" edges inside them.
    by_category: dict[str, list[str]] = {}
    for node in graph.of_kind("series"):
        category = node.meta.get("category")
        if not category:
            continue
        if category not in graph.nodes:
            graph.add_node(
                Node(
                    key=f"category:{category}",
                    kind="category",
                    label=category,
                    description=f"Indicator category: {category}.",
                )
            )
        graph.add_edge(node.key, f"category:{category}", "in_category")
        by_category.setdefault(category, []).append(node.key)

    for members in by_category.values():
        for left in members:
            for right in members:
                if left < right:
                    graph.add_edge(left, right, "compared_to")

    _add_themes(graph)

    if model:
        _add_metrics(graph, model)

    return graph


def _add_themes(graph: Graph) -> None:
    """Add theme nodes so everyday phrasing reaches the data.

    Edges are weighted by member rank: the first member of a theme is its most
    representative series, so "the economy" lands on GDP growth before it lands
    on manufacturing output.
    """
    for key, theme in THEMES.items():
        node_key = f"theme:{key}"
        graph.add_node(
            Node(
                key=node_key,
                kind="theme",
                label=theme["label"],
                description=theme["description"],
                synonyms=list(theme["synonyms"]),
            )
        )
        for rank, member in enumerate(theme["members"]):
            if member not in graph.nodes:
                continue
            weight = 1.0 if rank == 0 else (0.8 if rank < 3 else 0.6)
            graph.add_edge(node_key, member, "about_topic", weight=weight)


def _model_scope(model: dict) -> dict:
    """Return the inner semantic model, accepting either published shape.

    ``semantic.build_model()`` emits the released 0.1.1 revision, which wraps
    everything in a ``semantic_model`` array. Reading ``model["datasets"]``
    directly therefore found nothing and the whole metric layer silently
    vanished — a graph that looked fine but was missing exactly the cross-series
    knowledge worth having.
    """
    inner = model.get("semantic_model")
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        return inner[0]
    return model


def _add_metrics(graph: Graph, model: dict) -> None:
    """Add derived-metric nodes and wire them to the series they consume.

    Only *model-level* metrics are used: those are the ones that genuinely span
    series (``real_policy_rate`` = repo less CPI). The 170 dataset-level metrics
    are per-series aggregates like ``value_average`` — they would add one node
    per series per aggregate and make retrieval noisier without teaching the
    chatbot anything it cannot get from the tool layer.

    Metric SQL is not parsed. Instead the referenced ``fct_<series>`` dataset
    names are extracted, which is exact for the expressions this project
    generates and does not pretend to be a general SQL parser.
    """
    scope = _model_scope(model)

    for metric in scope.get("metrics") or []:
        name = metric.get("name")
        if not name:
            continue
        sql = (metric.get("expression") or {}).get("sql", "")
        # fct_<series> references, minus any self-reference.
        inputs = [
            key
            for key in dict.fromkeys(m.group(1) for m in re.finditer(r"fct_([a-z0-9_]+)", sql))
            if key in graph.nodes
        ]
        if not inputs:
            continue

        ai_context = metric.get("ai_context") or {}
        graph.add_node(
            Node(
                key=f"metric:{name}",
                kind="metric",
                label=name.replace("_", " "),
                description=metric.get("description") or "",
                synonyms=list(ai_context.get("synonyms") or []),
                meta={
                    "sql": sql,
                    "inputs": inputs,
                    "examples": list(ai_context.get("examples") or []),
                },
            )
        )
        for key in inputs:
            # Both directions: asked about the series, the metric explains it;
            # asked about the metric, the series is what to plot.
            graph.add_edge(f"metric:{name}", key, "derived_from")
            graph.add_edge(key, f"metric:{name}", "measured_by")
