from fastapi import APIRouter
from app.api.auth import router as auth_router
from app.api.patients import router as patients_router
from app.api.reports import router as reports_router
from app.api.lab_results import router as lab_results_router
from app.api.rag import router as rag_router
from app.api.assistant import router as assistant_router
from app.api.xai import router as xai_router
from app.api.mri import router as mri_router
from app.api.progression import router as progression_router
from app.api.segmentation import router as segmentation_router

api_router = APIRouter(prefix="/api")
api_router.include_router(auth_router)
api_router.include_router(patients_router)
api_router.include_router(reports_router)
api_router.include_router(lab_results_router)
api_router.include_router(rag_router)
api_router.include_router(assistant_router)
api_router.include_router(xai_router)
api_router.include_router(mri_router)
api_router.include_router(progression_router)
api_router.include_router(segmentation_router)

__all__ = ["api_router"]
