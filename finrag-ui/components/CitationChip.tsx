import type { Citation } from "@/lib/types";

interface CitationChipProps {
  citation: Citation;
  onClick: () => void;
  index: number;
}

export default function CitationChip({ citation, onClick, index }: CitationChipProps) {
  return (
    <button
      onClick={onClick}
      className="font-mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        fontSize: "0.68rem",
        fontWeight: 500,
        padding: "2px 7px",
        borderRadius: 4,
        background: "var(--citation-bg)",
        color: "var(--citation-text)",
        border: "1px solid rgba(147,197,253,0.2)",
        cursor: "pointer",
        verticalAlign: "middle",
        margin: "0 2px",
        transition: "background 150ms",
        whiteSpace: "nowrap",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = "#254d73";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = "var(--citation-bg)";
      }}
      title={`${citation.ticker} · ${citation.filing_type} · ${citation.section} · p.${citation.page}`}
    >
      <span style={{ opacity: 0.6 }}>[{index}]</span>
      <span>{citation.ticker}</span>
      <span style={{ opacity: 0.5 }}>·</span>
      <span>{citation.filing_type}</span>
      <span style={{ opacity: 0.5 }}>·</span>
      <span style={{ maxWidth: 80, overflow: "hidden", textOverflow: "ellipsis" }}>
        {citation.section}
      </span>
      <span style={{ opacity: 0.5 }}>·</span>
      <span>p.{citation.page}</span>
    </button>
  );
}
