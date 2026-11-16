import type { CitationGroup, Verification } from "./api";

/**
 * Evidence rendering.
 *
 * The point of this interface is that a reader can check a figure without
 * leaving the page, so citations are visible by default rather than behind a
 * disclosure. Each item is labelled by kind, because "CPI was 4.5% in September"
 * and "CPI is forecast at 4.5% next September" are very different claims and a
 * reader must be able to tell them apart at a glance.
 */

const SOURCE_CLASS: Record<string, string> = {
  observation: "cite cite--observed",
  derived: "cite cite--derived",
  forecast: "cite cite--forecast",
};

export function Citations({ groups }: { groups: CitationGroup[] }) {
  if (!groups.length) return null;

  return (
    <div className="citations">
      {groups.map((group) => (
        <div className="cite-group" key={group.series_key}>
          <div className="cite-group__label">{group.label}</div>
          <div className="cite-group__items">
            {group.items.map((item, index) => (
              <span
                className={SOURCE_CLASS[item.source] ?? "cite"}
                key={`${group.series_key}-${item.source}-${item.period}-${index}`}
                title={`${item.source_label} · ${item.period}`}
              >
                <b>{item.value_text}</b>
                {item.detail ? <span className="cite__unit">{item.detail}</span> : null}
                <span className="cite__kind">{item.source_label}</span>
                <span className="cite__period">{item.period}</span>
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export function VerificationBanner({ verification }: { verification: Verification | null }) {
  if (!verification) return null;

  // The state comes from the server, so a new state cannot silently render as
  // "verified" here — unknown states fall back to a neutral warning style.
  const tone =
    verification.state === "verified"
      ? "verify verify--ok"
      : verification.state === "rejected"
        ? "verify verify--warn"
        : "verify verify--neutral";

  return (
    <div className={tone} role="status">
      <span className="verify__headline">{verification.headline}</span>
      <span className="verify__detail">{verification.detail}</span>
    </div>
  );
}
