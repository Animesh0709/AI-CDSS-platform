from pydantic import BaseModel, Field, ConfigDict

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    time_in_hospital: float = Field(3, ge=1, le=30)
    num_lab_procedures: float = Field(40, ge=0, le=200)
    num_procedures: float = Field(0, ge=0, le=20)
    num_medications: float = Field(12, ge=0, le=100)
    number_outpatient: float = Field(0, ge=0, le=100)
    number_emergency: float = Field(0, ge=0, le=100)
    number_inpatient: float = Field(0, ge=0, le=100)
    number_diagnoses: float = Field(5, ge=1, le=30)


class PredictionOut(BaseModel):
    probability: float
    high_risk: bool = False
    decision_threshold: float = 0.5
    model_label: str
    base_value: float = 0.0
    patient_features: dict = Field(default_factory=dict)
    increasing_factors: list[dict]
    decreasing_factors: list[dict]
    lime_increasing_factors: list[dict] = []
    lime_decreasing_factors: list[dict] = []
    data_quality: list[str]
    ood_warning: str | None = None
    disclaimer: str
