import type { PipelineStage } from "@/lib/types";

interface StatusMessageProps {
  stage: PipelineStage | null;
  isLoading: boolean;
}

export default function StatusMessage({ stage, isLoading }: StatusMessageProps) {
  if (!isLoading || !stage) return null;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 0",
      }}
    >
      <span
        className="pulse-dot"
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "var(--accent)",
          flexShrink: 0,
          display: "inline-block",
        }}
      />
      <span
        className="font-mono"
        style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}
      >
        {stage}
      </span>
    </div>
  );
}
