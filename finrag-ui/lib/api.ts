import type { Citation, QueryFilters, QueryResponse, PipelineStage } from "./types";
import { PIPELINE_STAGES } from "./constants";

// Use a same-origin Next.js proxy route to avoid CORS + IPv6/IPv4 issues.
// The proxy forwards to the FastAPI backend server-side.
const PROXY_ENDPOINT = "/api/query";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

// Set to true to simulate SSE without a live backend
// CONFIG — set to false before connecting to live backend (Section 12)
const MOCK_MODE = false;

// ─── Health Check ──────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8002";
    const res = await fetch(`${API_URL}/healthz`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === "ok" || data.status === "healthy";
  } catch {
    return false;
  }
}

// ─── Stream Query ──────────────────────────────────────────────────────────

interface StreamCallbacks {
  onStage: (stage: PipelineStage) => void;
  onToken: (token: string) => void;
  onCitation: (citation: Citation) => void;
  onComplete: (response: QueryResponse) => void;
  onDecline: (reason: string) => void;
  onError: (error: string) => void;
}

export function streamQuery(
  query: string,
  filters: QueryFilters,
  callbacks: StreamCallbacks
): void {
  if (MOCK_MODE) {
    runMockStream(query, filters, callbacks);
    return;
  }
  runLiveStream(query, filters, callbacks);
}

// ─── Live SSE stream ───────────────────────────────────────────────────────

function runLiveStream(
  query: string,
  filters: QueryFilters,
  callbacks: StreamCallbacks
): void {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);

  // Build metadata_filter from UI filters to match backend ChromaDB schema
  const metadata_filter: Record<string, string> = {
    ticker: filters.ticker,
    form_type: filters.filing_type,
  };

  fetch(PROXY_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Auth is handled server-side by the Next.js proxy — no key needed in browser
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ query, metadata_filter }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (res.status === 401) {
        callbacks.onError("Invalid API key. Check NEXT_PUBLIC_API_KEY in .env.local.");
        return;
      }
      if (!res.ok) {
        callbacks.onError(`Backend returned ${res.status}: ${res.statusText}`);
        return;
      }
      if (!res.body) {
        callbacks.onError("No response body from backend.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Accumulated state across events
      let accumulatedAnswer = "";
      const accumulatedCitations: Citation[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE blocks are separated by double newlines
        // Each block may have: "event: <name>\ndata: <json>\n"
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          if (!block.trim()) continue;

          const lines = block.split(/\r?\n/);
          let eventName = "";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataStr = line.slice(5).trim();
            }
          }

          if (!dataStr) continue;

          let data: Record<string, unknown>;
          try {
            data = JSON.parse(dataStr);
          } catch {
            continue; // Malformed JSON — skip silently
          }

          handleNamedEvent(
            eventName,
            data,
            callbacks,
            accumulatedCitations,
            (token) => { accumulatedAnswer += token; },
            () => accumulatedAnswer
          );
        }
      }
    })
    .catch((err: Error) => {
      if (err.name === "AbortError") {
        callbacks.onError("Request timed out after 120 seconds.");
      } else {
        callbacks.onError(`Connection failed: ${err.message}`);
      }
    })
    .finally(() => clearTimeout(timeout));
}

// Map backend stage events to our PipelineStage union type
const STAGE_MAP: Record<string, PipelineStage> = {
  retrieval_start:  "Encoding query...",
  chunks_found:     "Retrieving candidate chunks...",
  rerank_done:      "Reranking with cross-encoder...",
  generation_start: "Generating grounded answer...",
};

function handleNamedEvent(
  eventName: string,
  data: Record<string, unknown>,
  callbacks: StreamCallbacks,
  accumulatedCitations: Citation[],
  appendToken: (t: string) => void,
  getAnswer: () => string
): void {
  // Pipeline stage events
  if (eventName in STAGE_MAP) {
    callbacks.onStage(STAGE_MAP[eventName]);

    // Inject the "Enforcing citations..." stage between rerank and generation
    if (eventName === "rerank_done") {
      setTimeout(() => callbacks.onStage("Enforcing citations..."), 150);
    }
    return;
  }

  // Streaming answer tokens
  if (eventName === "answer_chunk") {
    const token = (data.text as string) ?? "";
    if (token) {
      appendToken(token);
      callbacks.onToken(token);
    }
    return;
  }

  // Citation event — map backend shape to frontend Citation interface
  if (eventName === "citation") {
    const citation = mapBackendCitation(data);
    if (citation) {
      accumulatedCitations.push(citation);
      callbacks.onCitation(citation);
    }
    return;
  }

  // Stream complete
  if (eventName === "done") {
    const route = (data.route as string) ?? "";
    const isValid = data.is_valid as boolean;

    // Treat stub/blocked/decline routes as declined
    const declined =
      !isValid ||
      route === "decline" ||
      route === "blocked" ||
      route === "stub";

    if (declined) {
      const reason =
        route === "blocked"
          ? "This query was blocked by the content filter."
          : route === "stub"
          ? "Pipeline not initialized — no data loaded yet. Start the backend with FINRAG_INIT_PIPELINE=true."
          : "Insufficient evidence found in the selected filing.";
      callbacks.onDecline(reason);
    }

    callbacks.onComplete({
      answer: getAnswer(),
      citations: accumulatedCitations,
      confidence: isValid ? 0.85 : 0.0,
      declined,
      decline_reason: declined
        ? "See decline reason above."
        : null,
    });
    return;
  }

  // Error event
  if (eventName === "error") {
    callbacks.onError((data.error as string) ?? "Unknown backend error.");
  }
}

// Map backend CitationResponse fields → frontend Citation interface
function mapBackendCitation(raw: Record<string, unknown>): Citation | null {
  const chunkId = (raw.chunk_id as string) ?? "";
  const filingRef = (raw.filing_reference as string) ?? "";
  const section = (raw.section as string) ?? "";
  const page = (raw.page as number) ?? 0;
  const text = (raw.text as string) ?? (raw.chunk_text as string) ?? "";

  // Parse "AAPL 10-K" style filing_reference into ticker + filing_type
  const parts = filingRef.trim().split(/\s+/);
  const ticker = parts[0] || "N/A";
  const filingType = parts[1] || "N/A";

  if (!chunkId && !filingRef) return null;

  return {
    chunk_id: chunkId,
    ticker,
    filing_type: filingType,
    section,
    page,
    text: text || `Source: ${filingRef} — ${section}`,
  };
}

// ─── Mock SSE stream ───────────────────────────────────────────────────────

const MOCK_ANSWER =
  "For fiscal year 2025, Apple reported total net sales of $391.035 billion [1]. " +
  "Net income was $93.736 billion [2]. " +
  "Services net sales reached $96.169 billion, representing approximately 25% of total revenue [3].";

const MOCK_CITATIONS: Citation[] = [
  {
    chunk_id: "aapl_rev_001",
    ticker: "AAPL",
    filing_type: "10-K",
    section: "MD&A",
    page: 42,
    text: "Total net sales were $391.035 billion for fiscal 2025, an increase of 2% compared to fiscal 2024. iPhone sales led growth at $233.1 billion.",
  },
  {
    chunk_id: "aapl_net_002",
    ticker: "AAPL",
    filing_type: "10-K",
    section: "Consolidated Statements of Operations",
    page: 51,
    text: "Net income $93,736 million for the year ended September 27, 2025. Earnings per diluted share were $6.42.",
  },
  {
    chunk_id: "aapl_svc_003",
    ticker: "AAPL",
    filing_type: "10-K",
    section: "MD&A — Services",
    page: 44,
    text: "Services net sales increased to $96.169 billion during fiscal 2025, driven by growth in App Store, Apple Music, and iCloud subscriptions.",
  },
];

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function runMockStream(
  _query: string,
  _filters: QueryFilters,
  callbacks: StreamCallbacks
): Promise<void> {
  // Cycle through pipeline stages
  for (const stage of PIPELINE_STAGES) {
    await delay(400);
    callbacks.onStage(stage);
  }

  await delay(300);

  // Stream answer token by token (word by word)
  const words = MOCK_ANSWER.split(" ");
  for (const word of words) {
    await delay(40);
    callbacks.onToken(word + " ");
  }

  // Emit citations
  for (const citation of MOCK_CITATIONS) {
    await delay(100);
    callbacks.onCitation(citation);
  }

  await delay(200);

  callbacks.onComplete({
    answer: MOCK_ANSWER,
    citations: MOCK_CITATIONS,
    confidence: 0.94,
    declined: false,
    decline_reason: null,
  });
}
