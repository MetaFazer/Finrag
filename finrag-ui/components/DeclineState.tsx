interface DeclineStateProps {
  ticker: string;
  filingType: string;
  period: string;
  declineReason: string | null;
  onReset: () => void;
}

export default function DeclineState({
  ticker,
  filingType,
  period,
  declineReason,
  onReset,
}: DeclineStateProps) {
  return (
    <div
      className="fade-in"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        padding: "40px 48px",
        textAlign: "center",
      }}
    >
      <div
        style={{
          background: "var(--decline-bg)",
          border: "1px solid var(--decline-border)",
          borderRadius: 12,
          padding: "36px 40px",
          maxWidth: 520,
          width: "100%",
        }}
      >
        {/* Shield icon */}
        <div style={{ marginBottom: 20 }}>
          <svg
            width="36"
            height="36"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            style={{ margin: "0 auto", display: "block" }}
          >
            <path
              d="M12 2L3 6v6c0 5.25 3.75 10.15 9 11.25C17.25 22.15 21 17.25 21 12V6l-9-4z"
              stroke="#7f1d1d"
              strokeWidth="1.5"
              fill="rgba(127,29,29,0.15)"
            />
            <path d="M12 8v4M12 16h.01" stroke="var(--decline-text)" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>

        <h2
          style={{
            fontSize: "1.05rem",
            fontWeight: 600,
            color: "var(--decline-text)",
            marginBottom: 12,
          }}
        >
          No Sufficient Evidence Found
        </h2>

        <p
          style={{
            fontSize: "0.85rem",
            color: "var(--text-secondary)",
            lineHeight: 1.7,
            marginBottom: declineReason ? 20 : 0,
          }}
        >
          FinRAG reviewed the selected{" "}
          <span className="font-mono" style={{ color: "var(--citation-text)", fontSize: "0.8rem" }}>
            {ticker} {filingType}
          </span>{" "}
          filing for{" "}
          <span className="font-mono" style={{ color: "var(--citation-text)", fontSize: "0.8rem" }}>
            {period}
          </span>{" "}
          and could not find chunks that adequately support an answer to your question.
        </p>

        {declineReason && (
          <blockquote
            style={{
              borderLeft: "2px solid var(--decline-border)",
              paddingLeft: 14,
              margin: "0 0 20px",
              fontSize: "0.8rem",
              color: "var(--decline-text)",
              fontStyle: "italic",
              textAlign: "left",
            }}
          >
            {declineReason}
          </blockquote>
        )}

        <div
          style={{
            borderTop: "1px solid var(--decline-border)",
            paddingTop: 18,
            marginTop: declineReason ? 0 : 20,
          }}
        >
          <p
            style={{
              fontSize: "0.78rem",
              color: "var(--text-muted)",
              lineHeight: 1.65,
              marginBottom: 18,
            }}
          >
            This is by design. FinRAG is built to decline rather than generate unsupported answers.
            Try rephrasing your question or selecting a different filing period.
          </p>
          <button
            onClick={onReset}
            style={{
              background: "rgba(127,29,29,0.2)",
              border: "1px solid var(--decline-border)",
              borderRadius: 6,
              color: "var(--decline-text)",
              fontSize: "0.8rem",
              padding: "8px 18px",
              cursor: "pointer",
              transition: "background 150ms",
            }}
            onMouseEnter={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.background = "rgba(127,29,29,0.35)")
            }
            onMouseLeave={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.background = "rgba(127,29,29,0.2)")
            }
          >
            Try a different question
          </button>
        </div>
      </div>
    </div>
  );
}
