import type {
  TickerOption,
  FilingTypeOption,
  ExampleQuery,
  PipelineStage,
} from "./types";

export const TICKERS: TickerOption[] = [
  { value: "AAPL", label: "AAPL", fullName: "Apple Inc." },
  { value: "MSFT", label: "MSFT", fullName: "Microsoft Corporation" },
  { value: "TSLA", label: "TSLA", fullName: "Tesla, Inc." },
  { value: "GOOGL", label: "GOOGL", fullName: "Alphabet Inc." },
  { value: "AMZN", label: "AMZN", fullName: "Amazon.com, Inc." },
];

export const FILING_TYPES: FilingTypeOption[] = [
  { value: "10-K", label: "10-K", description: "Annual Report" },
  { value: "10-Q", label: "10-Q", description: "Quarterly Report" },
  { value: "8-K", label: "8-K", description: "Current Report" },
];

export const FISCAL_PERIODS: string[] = [
  "FY2025",
  "FY2024",
  "Q3 2025",
  "Q2 2025",
  "Q1 2025",
  "Q4 2024",
  "Q3 2024",
];

export const EXAMPLE_QUERIES: ExampleQuery[] = [
  {
    query: "What was the total revenue and net income?",
    description: "Revenue & net income summary",
  },
  {
    query: "What AI-related risk factors were disclosed?",
    description: "AI & technology risk factors",
  },
  {
    query: "How did operating margin change year over year?",
    description: "Operating margin YoY change",
  },
  {
    query: "What did management say about future guidance?",
    description: "Management guidance & outlook",
  },
  {
    query: "Were there any goodwill impairments or write-downs?",
    description: "Goodwill impairments & write-downs",
  },
  {
    query: "What were the primary drivers of revenue growth?",
    description: "Revenue growth drivers",
  },
];

export const PIPELINE_STAGES: PipelineStage[] = [
  "Encoding query...",
  "Retrieving candidate chunks...",
  "Reranking with cross-encoder...",
  "Enforcing citations...",
  "Generating grounded answer...",
];

export const DEFAULT_FILTERS = {
  ticker: "AAPL",
  filing_type: "10-K",
  fiscal_period: "FY2025",
};
