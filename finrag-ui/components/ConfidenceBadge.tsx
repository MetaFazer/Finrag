import type { ConfidenceLevel } from "@/lib/types";

interface ConfidenceBadgeProps {
  confidence: number | null;
  declined: boolean;
}

function getLevel(confidence: number | null, declined: boolean): ConfidenceLevel {
  if (declined) return "declined";
  if (confidence === null) return "low";
  if (confidence > 0.8) return "high";
  if (confidence >= 0.5) return "medium";
  return "low";
}

const LEVEL_STYLES: Record<ConfidenceLevel, { bg: string; border: string; text: string }> = {
  high:     { bg: "rgba(34,197,94,0.12)",  border: "rgba(34,197,94,0.35)",  text: "var(--confidence-high)" },
  medium:   { bg: "rgba(234,179,8,0.12)",  border: "rgba(234,179,8,0.35)",  text: "var(--confidence-mid)" },
  low:      { bg: "rgba(239,68,68,0.12)",  border: "rgba(239,68,68,0.35)",  text: "var(--confidence-low)" },
  declined: { bg: "rgba(239,68,68,0.12)",  border: "rgba(239,68,68,0.35)",  text: "var(--confidence-low)" },
};

export default function ConfidenceBadge({ confidence, declined }: ConfidenceBadgeProps) {
  const level = getLevel(confidence, declined);
  const { bg, border, text } = LEVEL_STYLES[level];

  const label =
    declined
      ? "Insufficient Evidence"
      : confidence === null
      ? "—"
      : confidence.toFixed(2);

  return (
    <span
      className="font-mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        fontSize: "0.7rem",
        fontWeight: 500,
        padding: "3px 9px",
        borderRadius: 20,
        background: bg,
        border: `1px solid ${border}`,
        color: text,
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          width: 5,
          height: 5,
          borderRadius: "50%",
          background: text,
          flexShrink: 0,
        }}
      />
      {!declined && confidence !== null && (
        <span style={{ color: "var(--text-muted)", marginRight: 1 }}>
          {level === "high" ? "High" : level === "medium" ? "Med" : "Low"}
        </span>
      )}
      {label}
    </span>
  );
}
