from pydantic import BaseModel, Field


class AssistantQuestion(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    patient_id: int | None = Field(default=None, gt=0)
    limit: int = Field(default=4, ge=1, le=8)


class PatientContextOut(BaseModel):
    patient_id: int
    patient_name: str
    history: list[str]
    report_matches: list[str]


class AssistantAnswerOut(BaseModel):
    question: str
    status: str
    answer: str | None = None
    message: str | None = None
    evidence: list[dict]
    patient_context: PatientContextOut | None = None


class ClinicalDraftOut(BaseModel):
    patient_id: int
    draft: str
    report_count: int
    disclaimer: str
    summary_mode: str | None = None
    generated_at: str | None = None
