"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import QueryPanel from "@/components/QueryPanel";
import CitationDrawer from "@/components/CitationDrawer";
import type { Citation, QueryFilters } from "@/lib/types";
import { DEFAULT_FILTERS } from "@/lib/constants";

export default function Home() {
  // Filter state — passed to both Sidebar (for editing) and QueryPanel (for querying)
  const [filters, setFilters] = useState<QueryFilters>(DEFAULT_FILTERS);

  // Citation drawer state — lives at page level so it overlays everything
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Example query selected from sidebar — threaded down to QueryPanel
  const [pendingQuery, setPendingQuery] = useState<string>("");

  const handleCitationClick = useCallback((citation: Citation) => {
    setSelectedCitation(citation);
    setDrawerOpen(true);
  }, []);

  const handleDrawerClose = useCallback(() => {
    setDrawerOpen(false);
    // Clear the citation after slide-out animation completes
    setTimeout(() => setSelectedCitation(null), 300);
  }, []);

  const handleExampleSelect = useCallback((query: string) => {
    setPendingQuery(query);
  }, []);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        height: "100vh",
        width: "100vw",
        overflow: "hidden",
        background: "var(--bg-base)",
      }}
    >
      {/* ── Sidebar: 280px fixed ──────────────────────────────── */}
      <Sidebar
        filters={filters}
        onFiltersChange={setFilters}
        onExampleSelect={handleExampleSelect}
      />

      {/* ── Main content: fills remaining space ──────────────── */}
      <QueryPanel
        filters={filters}
        onCitationClick={handleCitationClick}
        pendingQuery={pendingQuery}
        onPendingQueryConsumed={() => setPendingQuery("")}
      />

      {/* ── Citation drawer: page-level overlay ──────────────── */}
      <CitationDrawer
        citation={selectedCitation}
        isOpen={drawerOpen}
        onClose={handleDrawerClose}
      />
    </div>
  );
}
