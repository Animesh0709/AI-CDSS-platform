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
    compatibility update only covers the Week 3 field for existing student
    development databases created before clinical extraction was added.
    """
    if not database_url.startswith("sqlite"):
        return
    with engine.begin() as connection:
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(reports)"))}
        if columns and "clinical_entities" not in columns:
            connection.execute(text("ALTER TABLE reports ADD COLUMN clinical_entities TEXT"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
