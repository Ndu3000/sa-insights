/**
 * Transport and types for the chat API.
 *
 * Kept separate from the components so the rendering does not depend on how the
 * answer arrives. If the backend later grows a streaming agent protocol, this
 * file changes and the component tree does not.
 *
 * The shapes mirror `sa_insights/chat/api.py` exactly. `citation_groups` and
 * `verification` are computed server-side in `present.py`, so this client never
 * re-derives whether an answer was verified — two implementations of that rule
 * would eventually disagree, and the disagreement would be about trust.
 */

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ||
  "https://ndu3000--sa-insights-chat-api.us-east.modal.run";

export type AnswerKind = "answer" | "clarify" | "refuse";

export interface CitationItem {
  source: string;
  source_label: string;
  period: string;
  value: number | null;
  value_text: string;
  detail: string;
}

export interface CitationGroup {
  series_key: string;
  label: string;
  items: CitationItem[];
}

export interface Verification {
  state: string;
  headline: string;
  detail: string;
  figures: number;
}

export interface AskResponse {
  kind: AnswerKind;
  text: string;
  confidence: number;
  follow_up: string | null;
  unverified_numbers: number[];
  evidence_series: string[];
  citation_groups: CitationGroup[];
  verification: Verification | null;
}

export interface Health {
  status: string;
  series?: number;
  observations?: number;
  last_updated?: string;
  error?: string;
}

/** Raised for transport failures, so the UI can distinguish them from a refusal.
 *
 * A refusal is a successful answer that says "the data does not cover this".
 * A network error is not an answer at all, and showing it in the same style
 * would imply the bot had considered the question and declined.
 */
export class TransportError extends Error {}

export async function ask(
  question: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<AskResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
      signal,
    });
  } catch (cause) {
    if ((cause as Error)?.name === "AbortError") throw cause;
    throw new TransportError(
      "Could not reach the API. It may be starting up — try again in a moment.",
    );
  }

  if (!response.ok) {
    throw new TransportError(`The API returned ${response.status}.`);
  }
  return (await response.json()) as AskResponse;
}

export async function health(): Promise<Health> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return (await response.json()) as Health;
  } catch {
    return { status: "unreachable" };
  }
}

export async function reset(sessionId: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "reset", session_id: sessionId }),
    });
  } catch {
    // Clearing server context is best-effort; the local transcript is cleared
    // regardless, so a failure here must not block the user starting over.
  }
}

/** A stable per-tab id, so follow-ups resolve without identifying anyone.
 *
 * Deliberately not persisted to localStorage: this is a public endpoint with no
 * auth, and a durable identifier would let conversations be linked across
 * sessions for no benefit to the user.
 */
export function newSessionId(): string {
  const random =
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  return `web-${random}`;
}
