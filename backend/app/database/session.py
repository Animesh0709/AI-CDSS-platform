from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

database_url = settings.DATABASE_URL
connect_args = {}

if database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    database_url,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def apply_lightweight_schema_updates():
    """Keep the local SQLite development DB usable without a migration tool.

    Production deployments should use proper database migrations. This small
    compatibility update covers Week 3 fields and the new EWS tables.
    """
    if not database_url.startswith("sqlite"):
        return
    with engine.begin() as connection:
        # Week 3: clinical entities in reports
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(reports)"))}
        if columns and "clinical_entities" not in columns:
            connection.execute(text("ALTER TABLE reports ADD COLUMN clinical_entities TEXT"))

        # EWS: vital_signs table
        tables = {row[0] for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}

        if "vital_signs" not in tables:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS vital_signs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL REFERENCES patients(id),
                    doctor_id INTEGER NOT NULL REFERENCES users(id),
                    recorded_at DATETIME NOT NULL,
                    heart_rate FLOAT,
                    respiratory_rate FLOAT,
                    systolic_bp FLOAT,
                    diastolic_bp FLOAT,
                    spo2 FLOAT,
                    temperature_c FLOAT,
                    consciousness_level VARCHAR(10),
                    supplemental_o2 BOOLEAN DEFAULT 0,
                    urine_output_ml_hr FLOAT,
                    news2_score INTEGER,
                    news2_risk_level VARCHAR(20),
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))

        if "ews_assessments" not in tables:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS ews_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL REFERENCES patients(id),
                    doctor_id INTEGER NOT NULL REFERENCES users(id),
                    assessment_dt DATETIME NOT NULL,
                    ml_risk_score FLOAT,
                    ml_risk_label VARCHAR(20),
                    news2_score INTEGER,
                    shap_values_json TEXT,
                    trigger_features_json TEXT,
                    doctor_acknowledged BOOLEAN DEFAULT 0,
                    acknowledged_at DATETIME,
                    acknowledge_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))

        if "ecg_records" not in tables:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS ecg_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL REFERENCES patients(id),
                    doctor_id INTEGER NOT NULL REFERENCES users(id),
                    original_filename VARCHAR(255) NOT NULL,
                    stored_filename VARCHAR(255) NOT NULL,
                    file_path VARCHAR(1024) NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    mime_type VARCHAR(64) NOT NULL,
                    processing_status VARCHAR(50) NOT NULL DEFAULT 'uploaded',
                    analysis_status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    error_details TEXT,
                    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

