import { useEffect, useRef, useState } from "react";
import {
  API_BASE,
  TransportError,
  ask,
  health,
  newSessionId,
  reset,
  type AskResponse,
  type Health,
} from "./api";
import { Citations, VerificationBanner } from "./Evidence";

interface Turn {
  question: string;
  answer?: AskResponse;
  error?: string;
}

const STARTERS = [
  "What is CPI inflation now?",
  "How has the repo rate changed since 2020?",
  "What is GDP growth year to date?",
  "What is the unemployment rate?",
];

const FOLLOW_UPS = [
  "And unemployment?",
  "Compare both to today",
  "What about the rand?",
  "What do they confirm?",
];

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Health | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    health().then(setStatus);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, busy]);

  // Abort on unmount so a late response cannot set state on a dead component.
  useEffect(() => () => inFlight.current?.abort(), []);

  async function submit(question: string) {
    const text = question.trim();
    if (!text || busy) return;

    setDraft("");
    setBusy(true);
    setTurns((previous) => [...previous, { question: text }]);

    const controller = new AbortController();
    inFlight.current = controller;

    try {
      const answer = await ask(text, sessionId, controller.signal);
      setTurns((previous) =>
        previous.map((turn, index) =>
          index === previous.length - 1 ? { ...turn, answer } : turn,
        ),
      );
    } catch (cause) {
      if ((cause as Error)?.name === "AbortError") return;
      const message =
        cause instanceof TransportError
          ? cause.message
          : "Something went wrong asking that.";
      setTurns((previous) =>
        previous.map((turn, index) =>
          index === previous.length - 1 ? { ...turn, error: message } : turn,
        ),
      );
    } finally {
      setBusy(false);
      inFlight.current = null;
    }
  }

  async function clear() {
    inFlight.current?.abort();
    await reset(sessionId);
    setTurns([]);
    setSessionId(newSessionId());
    setBusy(false);
  }

  const suggestions = turns.length ? FOLLOW_UPS : STARTERS;

  return (
    <div className="app">
      <header className="brand">
        <div className="brand__mark">SA</div>
        <div className="brand__text">
          <h1>SA Insights</h1>
          <p>Ask the data · grounded answers with citations</p>
        </div>
        <div className="brand__meta">
          {status?.series ? (
            <>
              <b>{status.observations?.toLocaleString() ?? "—"}</b> observations
              <br />
              {status.series} series
            </>
          ) : (
            <span className="brand__pending">connecting…</span>
          )}
        </div>
      </header>

      <div className="guarantee">
        <span className="guarantee__title">Numbers come from the warehouse, not the model</span>
        <p>
          Every figure is fetched from stored observations, and each answer is checked
          before display: if a number cannot be traced to a stored value, the answer is
          rejected rather than shown.
        </p>
      </div>

      <main className="thread">
        {turns.length === 0 && !busy ? (
          <div className="empty">
            <b>Ask about the South African economy</b>
            <p>
              Inflation, the repo rate, GDP growth, unemployment, the rand — or start
              with one of the questions below.
            </p>
          </div>
        ) : null}

        {turns.map((turn, index) => (
          <article className="exchange" key={`${turn.question}-${index}`}>
            <div className="ask">
              <span className="ask__who">You</span>
              {turn.question}
            </div>

            {turn.error ? (
              <div className="answer answer--error">{turn.error}</div>
            ) : turn.answer ? (
              <>
                <div className={`answer answer--${turn.answer.kind}`}>
                  {turn.answer.text}
                </div>
                {turn.answer.kind === "answer" ? (
                  <Citations groups={turn.answer.citation_groups} />
                ) : null}
                {turn.answer.follow_up ? (
                  <div className="follow-up">↳ {turn.answer.follow_up}</div>
                ) : null}
                <VerificationBanner verification={turn.answer.verification} />
              </>
            ) : (
              <div className="answer answer--pending">Checking the data…</div>
            )}
          </article>
        ))}

        {busy ? <div className="thinking">Checking the warehouse…</div> : null}
        <div ref={endRef} />
      </main>

      <div className="chips">
        <span className="chips__label">
          {turns.length ? "Ask a follow-up" : "Try one of these"}
        </span>
        <div className="chips__row">
          {suggestions.map((option) => (
            <button
              className="chip"
              key={option}
              onClick={() => submit(option)}
              disabled={busy}
              type="button"
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit(draft);
        }}
      >
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Ask about the South African economy…"
          aria-label="Question"
          disabled={busy}
          maxLength={500}
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          {busy ? "…" : "Ask"}
        </button>
        {turns.length ? (
          <button type="button" className="ghost" onClick={clear} disabled={busy}>
            Clear
          </button>
        ) : null}
      </form>

      <footer className="foot">
        Forecasts are statistical extrapolations, not predictions of policy or events.
        <br />
        <a href={API_BASE + "/docs"} target="_blank" rel="noreferrer">
          API
        </a>
        {" · "}
        <a
          href="https://ndu3000--sa-insights-dashboard.us-east.modal.run"
          target="_blank"
          rel="noreferrer"
        >
          Full dashboard
        </a>
        {" · "}
        <a
          href="https://ndu3000--sa-insights-mlflow-server.us-east.modal.run"
          target="_blank"
          rel="noreferrer"
        >
          Model tracking
        </a>
      </footer>
    </div>
  );
}
