"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useFinRAGQuery } from "@/hooks/useFinRAGQuery";
import ConfidenceBadge from "./ConfidenceBadge";
import StatusMessage from "./StatusMessage";
import CitationChip from "./CitationChip";
import DeclineState from "./DeclineState";
import type { Citation, QueryFilters } from "@/lib/types";
import { EXAMPLE_QUERIES } from "@/lib/constants";

interface QueryPanelProps {
  filters: QueryFilters;
  onCitationClick: (citation: Citation) => void;
  pendingQuery?: string;
  onPendingQueryConsumed?: () => void;
}

function renderAnswerWithCitations(
  answer: string,
  citations: Citation[],
  onCitationClick: (c: Citation) => void,
  isLoading: boolean
): React.ReactNode {
  if (!answer) return null;

  // Replace [N] markers with CitationChip components
  const parts = answer.split(/(\[\d+\])/g);
  return (
    <span className={isLoading ? "cursor-blink" : ""}>
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          const idx = parseInt(match[1], 10);
          const citation = citations[idx - 1];
          if (citation) {
            return (
              <CitationChip
                key={i}
                citation={citation}
                index={idx}
                onClick={() => onCitationClick(citation)}
              />
            );
          }
          return <span key={i} className="font-mono" style={{ color: "var(--citation-text)", fontSize: "0.75rem" }}>{part}</span>;
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}

export default function QueryPanel({ filters, onCitationClick, pendingQuery, onPendingQueryConsumed }: QueryPanelProps) {
  const { submit, reset, answer, citations, confidence, isLoading, currentStage, declined, declineReason, error, hasResult } = useFinRAGQuery();
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  // Consume pending query from sidebar example clicks
  useEffect(() => {
    if (pendingQuery) {
      setQuery(pendingQuery);
      onPendingQueryConsumed?.();
      setTimeout(() => textareaRef.current?.focus(), 0);
    }
  }, [pendingQuery, onPendingQueryConsumed]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }, [query]);

  // Scroll answer area to bottom as tokens arrive
  useEffect(() => {
    if (answerRef.current && isLoading) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [answer, isLoading]);

  const handleSubmit = useCallback(() => {
    if (!query.trim() || isLoading) return;
    submit(query.trim(), filters);
  }, [query, isLoading, submit, filters]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleReset = () => {
    reset();
    setQuery("");
    textareaRef.current?.focus();
  };

  const scopeLabel = `${filters.ticker} · ${filters.filing_type} · ${filters.fiscal_period}`;

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
        background: "var(--bg-base)",
      }}
    >
      {/* ── Header bar ─────────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "14px 24px",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-panel)",
          flexShrink: 0,
        }}
      >
        <span
          className="font-mono"
          style={{ fontSize: "0.78rem", color: "var(--text-secondary)", letterSpacing: "0.04em" }}
        >
          {scopeLabel}
        </span>
        <ConfidenceBadge confidence={confidence} declined={declined} />
      </div>

      {/* ── Answer area ────────────────────────────────────────── */}
      <div
        ref={answerRef}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "0",
        }}
      >
        {/* STATE 1 — Empty */}
        {!hasResult && !isLoading && !error && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              padding: "40px 48px",
              gap: 40,
            }}
          >
            <div style={{ textAlign: "center" }}>
              <div
                className="font-mono"
                style={{
                  fontSize: "3.5rem",
                  fontWeight: 700,
                  color: "var(--text-muted)",
                  letterSpacing: "-0.04em",
                  opacity: 0.35,
                  lineHeight: 1,
                  marginBottom: 16,
                }}
              >
                FinRAG
              </div>
              <p
                style={{
                  fontSize: "0.95rem",
                  color: "var(--text-muted)",
                  lineHeight: 1.7,
                  maxWidth: 420,
                }}
              >
                Ask anything about SEC filings.
                <br />
                Every answer is grounded in evidence.
              </p>
            </div>

            {/* Example query tiles */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: 12,
                width: "100%",
                maxWidth: 740,
              }}
            >
              {EXAMPLE_QUERIES.slice(0, 3).map((eq, i) => (
                <button
                  key={i}
                  onClick={() => setQuery(eq.query)}
                  style={{
                    background: "var(--bg-panel)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    padding: "14px 16px",
                    textAlign: "left",
                    cursor: "pointer",
                    transition: "border-color 150ms, background 150ms",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-active)";
                    e.currentTarget.style.background = "var(--bg-elevated)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border)";
                    e.currentTarget.style.background = "var(--bg-panel)";
                  }}
                >
                  <div style={{ fontSize: "0.7rem", color: "var(--accent)", marginBottom: 6, fontWeight: 600 }}>
                    EXAMPLE
                  </div>
                  <div style={{ fontSize: "0.83rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    {eq.description}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* STATE 2 — Loading (streaming) */}
        {isLoading && (
          <div style={{ padding: "32px 40px" }}>
            <StatusMessage stage={currentStage} isLoading={isLoading} />
            {answer && (
              <div className="answer-text fade-in" style={{ marginTop: 20 }}>
                {renderAnswerWithCitations(answer, citations, onCitationClick, true)}
              </div>
            )}
          </div>
        )}

        {/* STATE 3 — Answer received */}
        {hasResult && !declined && !isLoading && (
          <div style={{ padding: "32px 40px" }}>
            <div className="answer-text fade-in">
              {renderAnswerWithCitations(answer, citations, onCitationClick, false)}
            </div>

            {/* Citations list */}
            {citations.length > 0 && (
              <div style={{ marginTop: 36 }}>
                <div className="text-label" style={{ marginBottom: 14 }}>
                  Sources
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {citations.map((c, i) => (
                    <div
                      key={c.chunk_id || i}
                      style={{
                        background: "var(--bg-panel)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        padding: "14px 16px",
                        transition: "border-color 150ms",
                        cursor: "pointer",
                      }}
                      onClick={() => onCitationClick(c)}
                      onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--border-active)")}
                      onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                          marginBottom: 8,
                        }}
                      >
                        <span
                          className="font-mono"
                          style={{
                            fontSize: "0.68rem",
                            background: "var(--citation-bg)",
                            color: "var(--citation-text)",
                            padding: "2px 7px",
                            borderRadius: 4,
                            border: "1px solid rgba(147,197,253,0.2)",
                          }}
                        >
                          [{i + 1}] {c.ticker} · {c.filing_type}
                        </span>
                        <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>
                          {c.section}
                        </span>
                        <span
                          className="font-mono"
                          style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginLeft: "auto" }}
                        >
                          p.{c.page}
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "var(--text-muted)",
                          lineHeight: 1.6,
                          overflow: "hidden",
                          display: "-webkit-box",
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: "vertical",
                        }}
                      >
                        {c.text.slice(0, 120)}
                        {c.text.length > 120 ? "…" : ""}
                      </div>
                      <div
                        style={{
                          marginTop: 8,
                          fontSize: "0.72rem",
                          color: "var(--accent)",
                        }}
                      >
                        View excerpt →
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* STATE 4 — Declined */}
        {hasResult && declined && !isLoading && (
          <DeclineState
            ticker={filters.ticker}
            filingType={filters.filing_type}
            period={filters.fiscal_period}
            declineReason={declineReason}
            onReset={handleReset}
          />
        )}

        {/* Error state */}
        {error && !isLoading && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              padding: "40px",
            }}
          >
            <div
              style={{
                background: "var(--decline-bg)",
                border: "1px solid var(--decline-border)",
                borderRadius: 8,
                padding: "20px 24px",
                maxWidth: 480,
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: "0.85rem", color: "var(--decline-text)", marginBottom: 12 }}>
                {error}
              </div>
              <button
                onClick={handleReset}
                style={{
                  background: "none",
                  border: "1px solid var(--decline-border)",
                  borderRadius: 5,
                  color: "var(--decline-text)",
                  fontSize: "0.78rem",
                  padding: "6px 14px",
                  cursor: "pointer",
                }}
              >
                Try again
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Status bar (loading only) ───────────────────────────── */}
      {isLoading && currentStage && (
        <div
          style={{
            padding: "6px 40px",
            borderTop: "1px solid var(--border)",
            flexShrink: 0,
          }}
        >
          <StatusMessage stage={currentStage} isLoading={isLoading} />
        </div>
      )}

      {/* ── Query input ─────────────────────────────────────────── */}
      <div
        style={{
          padding: "16px 24px 20px",
          borderTop: "1px solid var(--border)",
          background: "var(--bg-panel)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
          <div style={{ flex: 1, position: "relative" }}>
            <textarea
              ref={textareaRef}
              className="finrag-textarea"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about the selected filing..."
              rows={1}
              disabled={isLoading}
              style={{ display: "block", opacity: isLoading ? 0.6 : 1 }}
            />
            {query.length > 200 && (
              <div
                className="font-mono"
                style={{
                  position: "absolute",
                  bottom: 8,
                  right: 10,
                  fontSize: "0.65rem",
                  color: query.length > 500 ? "var(--confidence-low)" : "var(--text-muted)",
                }}
              >
                {query.length}
              </div>
            )}
          </div>

          <button
            onClick={handleSubmit}
            disabled={isLoading || !query.trim()}
            style={{
              background: isLoading || !query.trim() ? "var(--bg-elevated)" : "var(--accent)",
              border: "none",
              borderRadius: 8,
              color: isLoading || !query.trim() ? "var(--text-muted)" : "#fff",
              fontSize: "0.875rem",
              fontWeight: 600,
              padding: "0 20px",
              height: 44,
              cursor: isLoading || !query.trim() ? "not-allowed" : "pointer",
              transition: "background 200ms, color 200ms",
              whiteSpace: "nowrap",
              flexShrink: 0,
            }}
            onMouseEnter={(e) => {
              if (!isLoading && query.trim())
                e.currentTarget.style.background = "var(--accent-hover)";
            }}
            onMouseLeave={(e) => {
              if (!isLoading && query.trim())
                e.currentTarget.style.background = "var(--accent)";
            }}
          >
            {isLoading ? "Researching…" : "Research"}
          </button>
        </div>

        {/* Scope + keyboard hint */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 8,
          }}
        >
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
            Searching {filters.ticker} {filters.filing_type} {filters.fiscal_period}
          </span>
          <span style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
            ⌘ + Enter to submit
          </span>
        </div>
      </div>
    </div>
  );
}
