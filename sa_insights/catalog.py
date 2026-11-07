"""Catalogue of South African series we collect, and how to fetch each one.

Every entry declares its source. Nothing here is hard-coded to a value: the
collector resolves `fetch` against the matching provider in `sources.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Categories drive the dashboard navigation.
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Growth & Output",
    "Prices",
    "Labour",
    "Money & Rates",
    "External Sector",
    "Markets",
    "Government",
    "Population",
]

FREQ_LABELS = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}


@dataclass(frozen=True)
class SeriesSpec:
    """Declarative description of one macro series."""

    key: str
    label: str
    unit: str
    frequency: str  # "M" | "Q" | "A"
    category: str
    description: str
    # provider routing
    provider: str = "dbnomics"  # dbnomics | worldbank | fred
    fetch: dict = field(default_factory=dict)
    # presentation
    decimals: int = 1
    higher_is_better: bool | None = None
    optional: bool = False
    forecast: bool = True

    @property
    def freq_label(self) -> str:
        return FREQ_LABELS.get(self.frequency, self.frequency)

    @property
    def slug(self) -> str:
        return self.key


def _ifs(key, label, indicator, unit, category, description, freq="M", **kw):
    return SeriesSpec(
        key=key,
        label=label,
        unit=unit,
        frequency=freq,
        category=category,
        description=description,
        provider="dbnomics",
        fetch={
            "provider": "IMF",
            "dataset": "IFS",
            "dimensions": {"REF_AREA": ["ZA"], "INDICATOR": [indicator], "FREQ": [freq]},
        },
        **kw,
    )


def _wb(key, label, indicator, unit, category, description, **kw):
    return SeriesSpec(
        key=key,
        label=label,
        unit=unit,
        frequency="A",
        category=category,
        description=description,
        provider="worldbank",
        fetch={"indicator": indicator, "country": "ZAF"},
        **kw,
    )


def _sarb(key, label, code, unit, category, description, freq="M", start=None, **kw):
    """Series published by the South African Reserve Bank Web API.

    Preferred over the DBnomics IMF mirror wherever it exists: SARB is the
    primary publisher, so its figures are authoritative and current rather than
    a year behind.

    `start` overrides how far back to request. The API returns daily
    granularity for everything, so an unnecessarily early start just means a
    larger download.
    """
    fetch = {"timeseries_code": code}
    if start:
        fetch["start"] = start
    return SeriesSpec(
        key=key,
        label=label,
        unit=unit,
        frequency=freq,
        category=category,
        description=description,
        provider="sarb",
        fetch=fetch,
        **kw,
    )


def _weo(key, label, subject, unit_code, unit, category, description, **kw):
    return SeriesSpec(
        key=key,
        label=label,
        unit=unit,
        frequency="A",
        category=category,
        description=description,
        provider="dbnomics",
        fetch={
            "provider": "IMF",
            "dataset": "WEO",
            "dimensions": {"weo-country": ["ZAF"], "weo-subject": [subject]},
            "prefer_unit": unit_code,
        },
        **kw,
    )


SERIES: list[SeriesSpec] = [
    # ---------------------------------------------------------------- growth
    _wb(
        "gdp_growth",
        "GDP growth",
        "NY.GDP.MKTP.KD.ZG",
        "%",
        "Growth & Output",
        "Annual growth of real gross domestic product, constant 2015 prices.",
        higher_is_better=True,
    ),
    _wb(
        "gdp_per_capita",
        "GDP per capita",
        "NY.GDP.PCAP.KD",
        "US$ (2015)",
        "Growth & Output",
        "Real GDP per person, constant 2015 US dollars.",
        decimals=0,
        higher_is_better=True,
    ),
    _sarb(
        "retail_sales",
        "Retail trade",
        "MAN5008F",
        "index (2019=100)",
        "Growth & Output",
        "Volume of retail trade at constant prices. Source: SARB, updated monthly.",
        freq="M",
    ),
    _sarb(
        "manufacturing",
        "Manufacturing output",
        "MPR0000B",
        "index (2019=100)",
        "Growth & Output",
        "Manufacturing production index. Source: SARB, updated monthly.",
    ),
    _sarb(
        "mining",
        "Mining output",
        "PROB101B",
        "index (2019=100)",
        "Growth & Output",
        "Gold mining production volume index. Source: SARB, updated monthly.",
    ),
    _sarb(
        "vehicle_sales",
        "New vehicle sales",
        "MAN2022N",
        "index (2019=100)",
        "Growth & Output",
        "Total new vehicle sales index — a timely read on household demand.",
    ),
    _sarb(
        "leading_indicator",
        "Leading indicator",
        "DIFN003A",
        "index (2019=100)",
        "Growth & Output",
        "Composite leading business cycle indicator, designed to anticipate turning points.",
    ),
    SeriesSpec(
        key="real_gdp_q",
        label="Real GDP (quarterly)",
        unit="R million",
        frequency="Q",
        category="Growth & Output",
        description="Quarterly real GDP at constant prices, seasonally adjusted.",
        provider="fred",
        fetch={"series_id": "NGDPRSAXDCZAQ"},
        decimals=0,
        higher_is_better=True,
        optional=True,
    ),
    # ---------------------------------------------------------------- prices
    _sarb(
        "cpi_yoy",
        "CPI inflation (YoY)",
        "CPI1000F",
        "%",
        "Prices",
        "Consumer price inflation, year-on-year. Published by Stats SA and "
        "carried on the SARB Web API, which updates it within days of release.",
        decimals=2,
        higher_is_better=False,
        start="2000-01-01",
    ),
    _sarb(
        "ppi_yoy",
        "PPI inflation (YoY)",
        "PPI1000F",
        "%",
        "Prices",
        "Producer price inflation, year-on-year. Leads consumer inflation "
        "through the supply chain.",
        decimals=2,
        higher_is_better=False,
        start="2000-01-01",
    ),
    _ifs(
        "cpi_index",
        "CPI (all items)",
        "PCPI_IX",
        "index",
        "Prices",
        "Consumer price index level, all items. Base varies by year.",
        decimals=2,
    ),
    _wb(
        "inflation_annual",
        "Inflation (annual avg)",
        "FP.CPI.TOTL.ZG",
        "%",
        "Prices",
        "Annual average consumer price inflation.",
        higher_is_better=False,
    ),
    # ---------------------------------------------------------------- labour
    _wb(
        "unemployment",
        "Unemployment rate",
        "SL.UEM.TOTL.ZS",
        "%",
        "Labour",
        "Unemployed as a share of the total labour force (ILO modelled estimate).",
        higher_is_better=False,
    ),
    _wb(
        "lf_participation",
        "Labour force participation",
        "SL.TLF.CACT.ZS",
        "%",
        "Labour",
        "Labour force as a share of the population aged 15 and older.",
        higher_is_better=True,
    ),
    # ---------------------------------------------------------- money & rates
    _sarb(
        "policy_rate",
        "Policy rate",
        "MMRD002A",
        "%",
        "Money & Rates",
        "SARB policy rate (repo rate) set by the Monetary Policy Committee. "
        "Daily observations collapsed to month-end.",
        decimals=2,
        higher_is_better=False,
    ),
    _sarb(
        "lending_rate",
        "Prime lending rate",
        "MMRD000A",
        "%",
        "Money & Rates",
        "Prime overdraft lending rate. Source: SARB, updated with each MPC decision.",
        decimals=2,
        higher_is_better=False,
    ),
    _sarb(
        "bond_yield",
        "Long bond yield",
        "CMJD004A",
        "%",
        "Money & Rates",
        "Daily average yield on government bonds of 10 years and longer.",
        decimals=2,
        higher_is_better=False,
    ),
    # ------------------------------------------------------------ external
    _sarb(
        "usd_zar",
        "Rand per US dollar",
        "EXCX135D",
        "ZAR",
        "External Sector",
        "Official SARB reference rate, daily, collapsed to month-end. "
        "Higher means a weaker rand.",
        decimals=2,
        higher_is_better=False,
    ),
    _sarb(
        "reer",
        "Real effective exchange rate",
        "BOP5398M",
        "index (2022=100)",
        "External Sector",
        "Trade-weighted exchange rate adjusted for relative consumer prices. "
        "A rise means South African exports have become less competitive.",
        decimals=1,
        start="2000-01-01",
    ),
    _ifs(
        "exports_usd",
        "Exports (goods)",
        "TXG_FOB_USD",
        "US$ m",
        "External Sector",
        "Monthly value of merchandise exports, free on board.",
        decimals=0,
        higher_is_better=True,
    ),
    _ifs(
        "imports_usd",
        "Imports (goods)",
        "TMG_FOB_USD",
        "US$ m",
        "External Sector",
        "Monthly value of merchandise imports, free on board.",
        decimals=0,
    ),
    _wb(
        "current_account",
        "Current account balance",
        "BN.CAB.XOKA.GD.ZS",
        "% of GDP",
        "External Sector",
        "Current account balance as a share of GDP.",
    ),
    # ------------------------------------------------------------- markets
    _ifs(
        "equity_index",
        "Equity price index",
        "FPE_IX",
        "index",
        "Markets",
        "Share price index for the Johannesburg stock exchange.",
        decimals=1,
        higher_is_better=True,
    ),
    # ---------------------------------------------------------- government
    _wb(
        "govt_debt",
        "Gross government debt",
        "GC.DOD.TOTL.GD.ZS",
        "% of GDP",
        "Government",
        "Central government debt, total as a share of GDP.",
        higher_is_better=False,
    ),
    # ---------------------------------------------------------- population
    _wb(
        "population",
        "Population",
        "SP.POP.TOTL",
        "people",
        "Population",
        "Total population, mid-year estimate.",
        decimals=0,
        forecast=False,
    ),
    _wb(
        "internet_users",
        "Internet users",
        "IT.NET.USER.ZS",
        "% of population",
        "Population",
        "Individuals using the internet as a share of the population.",
        higher_is_better=True,
        forecast=False,
    ),
    # ------------------------------------------------ IMF WEO incl. forecasts
    _weo(
        "weo_growth",
        "IMF forecast: GDP growth",
        "NGDP_RPCH",
        "pcent",
        "%",
        "Growth & Output",
        "IMF World Economic Outlook growth path, including published projections.",
        higher_is_better=True,
        forecast=False,
    ),
    _weo(
        "weo_inflation",
        "IMF forecast: inflation",
        "PCPIPCH",
        "pcent",
        "%",
        "Prices",
        "IMF WEO inflation path, including published projections.",
        higher_is_better=False,
        forecast=False,
    ),
    _weo(
        "weo_unemployment",
        "IMF forecast: unemployment",
        "LUR",
        "pcent",
        "%",
        "Labour",
        "IMF WEO unemployment path, including published projections.",
        higher_is_better=False,
        forecast=False,
    ),
    _weo(
        "weo_govt_balance",
        "IMF forecast: fiscal balance",
        "GGXCNL_NGDP",
        "pcent",
        "% of GDP",
        "Government",
        "General government net lending/borrowing as a share of GDP.",
        forecast=False,
    ),
    _weo(
        "weo_govt_debt",
        "IMF forecast: government debt",
        "GGXWDG_NGDP",
        "pcent",
        "% of GDP",
        "Government",
        "General government gross debt as a share of GDP.",
        higher_is_better=False,
        forecast=False,
    ),
    _weo(
        "weo_current_account",
        "IMF forecast: current account",
        "BCA_NGDPD",
        "pcent",
        "% of GDP",
        "External Sector",
        "Current account balance as a share of GDP.",
        forecast=False,
    ),
]

BY_KEY: dict[str, SeriesSpec] = {s.key: s for s in SERIES}


def by_category() -> dict[str, list[SeriesSpec]]:
    out: dict[str, list[SeriesSpec]] = {c: [] for c in CATEGORIES}
    for spec in SERIES:
        out.setdefault(spec.category, []).append(spec)
    return out


# Series shown as headline KPIs on the overview tab, in order.
HEADLINE = [
    "gdp_growth",
    "cpi_yoy",
    "unemployment",
    "policy_rate",
    "usd_zar",
    "equity_index",
]

# Terse labels for KPI tiles, where horizontal space is scarce.
SHORT_LABELS = {
    "gdp_growth": "GDP growth",
    "gdp_per_capita": "GDP per head",
    "cpi_yoy": "Inflation",
    "cpi_index": "CPI",
    "inflation_annual": "Inflation",
    "unemployment": "Unemployment",
    "policy_rate": "Repo rate",
    "lending_rate": "Lending rate",
    "bond_yield": "Bond yield",
    "usd_zar": "Rand / USD",
    "equity_index": "JSE index",
    "weo_growth": "GDP growth",
    "weo_inflation": "Inflation",
    "weo_unemployment": "Unemployment",
    "weo_govt_balance": "Fiscal balance",
    "weo_govt_debt": "Govt debt",
    "weo_current_account": "Current acct",
    "govt_debt": "Govt debt",
    "current_account": "Current acct",
    "lf_participation": "Participation",
    "retail_sales": "Retail trade",
    "manufacturing": "Manufacturing",
    "mining": "Mining",
    "reer": "Real FX rate",
    "exports_usd": "Exports",
    "imports_usd": "Imports",
    "population": "Population",
    "internet_users": "Internet use",
    "real_gdp_q": "Real GDP",
}


def short_label(spec: SeriesSpec) -> str:
    return SHORT_LABELS.get(spec.key, spec.label)


# Default series for the forecast lab.
FORECAST_DEFAULT = "cpi_yoy"
