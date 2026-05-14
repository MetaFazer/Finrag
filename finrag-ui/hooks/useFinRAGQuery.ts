"use client";

import { useState, useCallback, useRef } from "react";
import { streamQuery } from "@/lib/api";
import type { Citation, QueryFilters, QueryResponse, PipelineStage } from "@/lib/types";

interface FinRAGQueryState {
  answer: string;
  citations: Citation[];
  confidence: number | null;
  isLoading: boolean;
  currentStage: PipelineStage | null;
  declined: boolean;
  declineReason: string | null;
  error: string | null;
  hasResult: boolean;
}

const INITIAL_STATE: FinRAGQueryState = {
  answer: "",
  citations: [],
  confidence: null,
  isLoading: false,
  currentStage: null,
  declined: false,
  declineReason: null,
  error: null,
  hasResult: false,
};

interface UseFinRAGQueryReturn extends FinRAGQueryState {
  submit: (query: string, filters: QueryFilters) => void;
  reset: () => void;
}

export function useFinRAGQuery(): UseFinRAGQueryReturn {
  const [state, setState] = useState<FinRAGQueryState>(INITIAL_STATE);
  const answerRef = useRef<string>("");

  const submit = useCallback((query: string, filters: QueryFilters) => {
    if (!query.trim()) return;

    // Reset to clean loading state
    answerRef.current = "";
    setState({
      ...INITIAL_STATE,
      isLoading: true,
    });

    streamQuery(query, filters, {
      onStage: (stage) => {
        setState((prev) => ({ ...prev, currentStage: stage }));
      },

      onToken: (token) => {
        answerRef.current += token;
        const snapshot = answerRef.current;
        setState((prev) => ({ ...prev, answer: snapshot }));
      },

      onCitation: (citation) => {
        setState((prev) => ({
          ...prev,
          citations: [...prev.citations, citation],
        }));
      },

      onComplete: (response: QueryResponse) => {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          currentStage: null,
          answer: response.answer || prev.answer,
          citations: response.citations?.length ? response.citations : prev.citations,
          confidence: response.confidence,
          declined: response.declined,
          declineReason: response.decline_reason,
          hasResult: true,
          error: null,
        }));
      },

      onDecline: (reason) => {
        setState((prev) => ({
          ...prev,
          declined: true,
          declineReason: reason,
        }));
      },

      onError: (error) => {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          currentStage: null,
          error,
          hasResult: false,
        }));
      },
    });
  }, []);

  const reset = useCallback(() => {
    answerRef.current = "";
    setState(INITIAL_STATE);
  }, []);

  return { ...state, submit, reset };
}
