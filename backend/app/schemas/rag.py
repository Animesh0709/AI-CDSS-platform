from pydantic import BaseModel, Field


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=4, ge=1, le=10)


class RagChunkOut(BaseModel):
    source: str
    source_url: str | None = None
    section: str | None = None
    text: str
    score: float


class RagSearchOut(BaseModel):
    query: str
    embedding_mode: str
    results: list[RagChunkOut]


class RagAnswerRequest(RagSearchRequest):
    """A clinician question answered only from retrieved local evidence."""


class RagAnswerOut(BaseModel):
    query: str
    status: str
    answer: str | None = None
    message: str | None = None
    provider: str | None = None
    model: str | None = None
    evidence: list[RagChunkOut]
