# SA Insights — chat front end (EXPERIMENTAL)

> **Status: experimental, not deployed.** The supported chat interface is the
> **Ask** tab in the Streamlit dashboard, which is live and has more of the
> evidence rendering wired up. This client is kept as a spike, not a product.
>
> **Known bug:** clicking a suggestion chip does not submit the question. The
> handler, CORS headers, API base URL and built bundle were all checked and are
> correct; the cause is still unidentified. Typing a question and pressing Ask
> is the path that has been exercised.
>
> **Not covered by the deploy gate.** `scripts/deploy.py` evaluates the Python
> chat path only, so nothing here is regression-tested. Treat it as unverified
> until that changes.

A small React client for the grounded Q&A API. Talks to the same
`sa_insights.chat.api` service the Streamlit tab uses, so the numbers, citations
and verification wording are identical in both.

## What has actually been verified

| Check | Result |
|-------|--------|
| `tsc --noEmit` | passes |
| `vite build` | 1.5 s; CSS 6.3 kB, JS 149 kB (48 kB gzip) |
| Renders in a browser | yes — brand bar, live counts, chips, composer |
| Reads `/health` | yes — showed 7,467 observations, 35 series |
| Suggestion chip submits | **no — known bug** |
| Deployed | no |

## Why plain React and not CopilotKit

The plan called for CopilotKit. Having read the API contract, it is the wrong
tool for this particular job, and the reason is worth writing down rather than
quietly substituting something else:

- **CopilotKit's value is agent orchestration** — shared state, generative UI,
  human-in-the-loop, tool-call rendering. This backend exposes none of that. It
  is one `POST /ask` returning a finished, already-verified answer. There is no
  agent loop for CopilotKit to drive.
- **Its TypeScript runtime expects an AG-UI event stream.** Ours returns a
  single JSON object. Bridging that means writing an adapter that translates a
  request/response call into a streaming protocol — real complexity added to
  make a framework fit, not to serve a user.
- **The interesting UI here is evidence, not chat.** Grouped citations,
  observation-versus-forecast labels, and an honest verification banner are
  bespoke components either way. CopilotKit would supply the text box and
  message list, which are the easy parts.

So this is ~400 lines of React with no framework lock-in. If the backend later
grows a real agent loop — streaming tool calls, intermediate state — CopilotKit
becomes the right choice and this component tree ports into it cleanly, because
the rendering is already separated from the transport in `api.ts`.

## Running

```bash
cd chat-ui
npm install
npm run dev          # http://localhost:5173
```

Point it at a different backend with `VITE_API_BASE`:

```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```

The default is the deployed Modal endpoint.

## Build

```bash
npm run build        # static bundle in dist/
npm run preview
```

`dist/` is plain static files, so it can be served from GitHub Pages, any CDN,
or a Modal static mount. Node is a build-time dependency only — it is
deliberately not part of the Python image.

## What it shows

| Element | Source |
|---|---|
| Answer text | `text` |
| Citation groups | `citation_groups` — computed server-side in `present.py` |
| Verification banner | `verification` — same wording as the Streamlit tab |
| Follow-up prompt | `follow_up` |
| Refusal / clarification styling | `kind` |

The grouping and verification strings come from the API rather than being
reimplemented here. Duplicating that logic in TypeScript is how two interfaces
start disagreeing about whether an answer was verified.
