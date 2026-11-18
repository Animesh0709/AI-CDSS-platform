import json

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from datetime import datetime
from sqlalchemy.orm import relationship
from app.database.session import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    report_type = Column(String(64), nullable=True)
    extracted_text = Column(Text, nullable=True)
    _clinical_entities = Column("clinical_entities", Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="reports")

    @property
    def clinical_entities(self):
        """Return stored NLP findings as JSON, never as a clinical conclusion."""
        if not self._clinical_entities:
            return None
        try:
            return json.loads(self._clinical_entities)
        except (TypeError, ValueError):
            return None

    @clinical_entities.setter
    def clinical_entities(self, value):
        self._clinical_entities = json.dumps(value) if value else None
