from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import get_current_user
from app.schemas.rag import RagAnswerOut, RagAnswerRequest, RagSearchOut, RagSearchRequest
from app.services.rag_retrieval import answer, rebuild_index, search

router = APIRouter(prefix="/rag", tags=["RAG Evidence Retrieval"])


@router.post("/search", response_model=RagSearchOut)
def search_guidelines(payload: RagSearchRequest, current_user=Depends(get_current_user)):
    """Retrieve local guideline excerpts only; no LLM answer is generated."""
    try:
        return search(payload.query, payload.limit)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error))


@router.post("/answer", response_model=RagAnswerOut)
def answer_from_guidelines(payload: RagAnswerRequest, current_user=Depends(get_current_user)):
    """Answer only when the curated retrieval evidence is sufficient and cited."""
    try:
        return answer(payload.query, payload.limit)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error))


@router.post("/reindex")
def reindex_guidelines(current_user=Depends(get_current_user)):
    try:
        return rebuild_index()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error))
