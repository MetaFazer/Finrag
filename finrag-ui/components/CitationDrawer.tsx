"use client";

import { useEffect, useRef } from "react";
import type { Citation } from "@/lib/types";

interface CitationDrawerProps {
  citation: Citation | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function CitationDrawer({ citation, isOpen, onClose }: CitationDrawerProps) {
  const drawerRef = useRef<HTMLDivElement>(null);

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const edgarUrl = citation
    ? `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${citation.ticker}&type=${citation.filing_type}&dateb=&owner=include&count=10`
    : "#";

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.55)",
          zIndex: 40,
          opacity: isOpen ? 1 : 0,
          pointerEvents: isOpen ? "auto" : "none",
          transition: "opacity 250ms ease",
        }}
      />

      {/* Drawer */}
      <div
        ref={drawerRef}
        style={{
          position: "fixed",
          top: 0,
          right: 0,
          width: 380,
          height: "100vh",
          background: "var(--bg-panel)",
          borderLeft: "1px solid var(--border)",
          zIndex: 50,
          display: "flex",
          flexDirection: "column",
          transform: isOpen ? "translateX(0)" : "translateX(100%)",
          transition: isOpen
            ? "transform 280ms cubic-bezier(0.22,1,0.36,1)"
            : "transform 220ms ease-in",
          overflowY: "auto",
        }}
      >
        {citation && (
          <>
            {/* Header */}
            <div
              style={{
                padding: "18px 20px 16px",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
              }}
            >
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span
                    className="font-mono"
                    style={{
                      fontSize: "0.95rem",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {citation.ticker}
                  </span>
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
                    {citation.filing_type}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                    maxWidth: 280,
                  }}
                >
                  {citation.section}
                </div>
                <div
                  className="font-mono"
                  style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}
                >
                  Page {citation.page}
                </div>
              </div>

              {/* Close button */}
              <button
                onClick={onClose}
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--text-muted)",
                  fontSize: "1.2rem",
                  cursor: "pointer",
                  padding: "0 4px",
                  lineHeight: 1,
                  transition: "color 150ms",
                }}
                onMouseEnter={(e) => ((e.target as HTMLElement).style.color = "var(--text-primary)")}
                onMouseLeave={(e) => ((e.target as HTMLElement).style.color = "var(--text-muted)")}
              >
                ×
              </button>
            </div>

            {/* Body — chunk text */}
            <div style={{ padding: "20px", flex: 1 }}>
              <div className="text-label" style={{ marginBottom: 12 }}>
                Source Excerpt
              </div>
              <blockquote
                style={{
                  background: "var(--bg-elevated)",
                  borderLeft: "3px solid var(--accent)",
                  borderRadius: "0 6px 6px 0",
                  padding: "14px 16px",
                  margin: 0,
                  fontSize: "0.875rem",
                  lineHeight: 1.75,
                  color: "var(--text-primary)",
                  fontStyle: "normal",
                  letterSpacing: "0.01em",
                }}
              >
                &ldquo;{citation.text}&rdquo;
              </blockquote>
            </div>

            {/* Footer — EDGAR link */}
            <div
              style={{
                padding: "14px 20px",
                borderTop: "1px solid var(--border)",
              }}
            >
              <a
                href={edgarUrl}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: "0.78rem",
                  color: "var(--accent)",
                  textDecoration: "none",
                  transition: "color 150ms",
                }}
                onMouseEnter={(e) => ((e.target as HTMLElement).style.color = "var(--accent-hover)")}
                onMouseLeave={(e) => ((e.target as HTMLElement).style.color = "var(--accent)")}
              >
                View original filing on SEC EDGAR
                <span style={{ fontSize: "0.9rem" }}>→</span>
              </a>
            </div>
          </>
        )}
      </div>
    </>
  );
}
