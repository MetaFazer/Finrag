"use client";

import { useEffect, useState, useRef } from "react";
import { checkHealth } from "@/lib/api";
import { TICKERS, FILING_TYPES, FISCAL_PERIODS, EXAMPLE_QUERIES } from "@/lib/constants";
import type { QueryFilters } from "@/lib/types";

interface SidebarProps {
  filters: QueryFilters;
  onFiltersChange: (filters: QueryFilters) => void;
  onExampleSelect: (query: string) => void;
}

export default function Sidebar({ filters, onFiltersChange, onExampleSelect }: SidebarProps) {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const check = async () => {
      const ok = await checkHealth();
      setBackendOnline(ok);
    };
    check();
    intervalRef.current = setInterval(check, 30_000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []);

  const selectedTicker = TICKERS.find((t) => t.value === filters.ticker);
  const selectedFilingType = FILING_TYPES.find((f) => f.value === filters.filing_type);

  return (
    <aside
      style={{
        width: 280,
        minWidth: 280,
        height: "100vh",
        background: "var(--bg-panel)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Logo */}
      <div style={{ padding: "20px 20px 16px", borderBottom: "1px solid var(--border)" }}>
        <div
          className="font-mono"
          style={{ fontSize: "1.375rem", fontWeight: 600, color: "var(--accent)", letterSpacing: "-0.02em" }}
        >
          FinRAG
        </div>
        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 3, letterSpacing: "0.05em" }}>
          SEC FILING RESEARCH
        </div>
      </div>

      {/* Filters */}
      <div style={{ padding: "16px 16px 0", flex: 1, overflowY: "auto" }}>
        <div className="text-label" style={{ marginBottom: 12 }}>Research Scope</div>

        {/* Company */}
        <div style={{ marginBottom: 14 }}>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: 6 }}>
            Company
          </label>
          <select
            className="finrag-select"
            value={filters.ticker}
            onChange={(e) => onFiltersChange({ ...filters, ticker: e.target.value })}
          >
            {TICKERS.map((t) => (
              <option key={t.value} value={t.value}>{t.fullName}</option>
            ))}
          </select>
        </div>

        {/* Filing Type */}
        <div style={{ marginBottom: 14 }}>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: 6 }}>
            Filing Type
          </label>
          <select
            className="finrag-select"
            value={filters.filing_type}
            onChange={(e) => onFiltersChange({ ...filters, filing_type: e.target.value })}
          >
            {FILING_TYPES.map((f) => (
              <option key={f.value} value={f.value}>{f.label} — {f.description}</option>
            ))}
          </select>
          {selectedFilingType && (
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 4 }}>
              {selectedFilingType.description}
            </div>
          )}
        </div>

        {/* Period */}
        <div style={{ marginBottom: 20 }}>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: 6 }}>
            Period
          </label>
          <select
            className="finrag-select"
            value={filters.fiscal_period}
            onChange={(e) => onFiltersChange({ ...filters, fiscal_period: e.target.value })}
          >
            {FISCAL_PERIODS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>

        {/* Divider */}
        <div style={{ borderTop: "1px solid var(--border)", marginBottom: 16 }} />

        {/* Example Queries */}
        <div className="text-label" style={{ marginBottom: 10 }}>Example Queries</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {EXAMPLE_QUERIES.map((eq, i) => (
            <button
              key={i}
              onClick={() => onExampleSelect(eq.query)}
              style={{
                background: "none",
                border: "none",
                color: "var(--text-secondary)",
                fontSize: "0.78rem",
                textAlign: "left",
                padding: "6px 8px",
                borderRadius: 5,
                cursor: "pointer",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                transition: "background 150ms, color 150ms",
              }}
              onMouseEnter={(e) => {
                (e.target as HTMLButtonElement).style.background = "var(--bg-elevated)";
                (e.target as HTMLButtonElement).style.color = "var(--text-primary)";
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLButtonElement).style.background = "none";
                (e.target as HTMLButtonElement).style.color = "var(--text-secondary)";
              }}
              title={eq.query}
            >
              {eq.description}
            </button>
          ))}
        </div>
      </div>

      {/* Backend status */}
      <div
        style={{
          padding: "12px 16px",
          borderTop: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background:
              backendOnline === null
                ? "var(--text-muted)"
                : backendOnline
                ? "var(--confidence-high)"
                : "var(--confidence-low)",
            flexShrink: 0,
          }}
        />
        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
          {backendOnline === null
            ? "Checking backend..."
            : backendOnline
            ? "Backend online"
            : "Backend offline — mock mode"}
        </span>
      </div>
    </aside>
  );
}
