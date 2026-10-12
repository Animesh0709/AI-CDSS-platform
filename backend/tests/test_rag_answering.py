"""Week 6 guardrails for local and optional LLM-backed evidence answers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import rag_retrieval  # noqa: E402


def test_known_question_returns_cited_extractive_evidence_without_llm(monkeypatch):
    monkeypatch.setattr(rag_retrieval.settings, "LLM_PROVIDER", "")
    monkeypatch.setattr(rag_retrieval.settings, "LLM_API_KEY", "")
    result = rag_retrieval.answer("What does the guideline say about diabetes screening?", 3)
    assert result["status"] == "evidence_summary"
    assert result["answer"]
    assert "[1]" in result["answer"]
    assert result["evidence"]


def test_unrelated_question_is_not_sent_to_an_llm(monkeypatch):
    monkeypatch.setattr(rag_retrieval.settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(rag_retrieval.settings, "LLM_API_KEY", "not-used")
    monkeypatch.setattr(rag_retrieval.settings, "LLM_MODEL", "not-used")
    result = rag_retrieval.answer("How do I repair a bicycle chain?", 3)
    assert result["status"] == "insufficient_evidence"
    assert result["answer"] is None
