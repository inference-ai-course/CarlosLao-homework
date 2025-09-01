"""
routers/health.py
-----------------
FastAPI router exposing health and readiness endpoints.
"""

from fastapi import APIRouter, Depends
from ..health import check_ffmpeg_installed, check_asr_ready, check_llm_ready, check_gtts_ready

router = APIRouter()

def get_services(request):
    return request.app.state.services

@router.get("/health")
async def health(services = Depends(get_services)):
    """
    Light liveness check of core components.
    """
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": services.asr.model is not None,
        "llm_ready": services.llm.generator is not None,
        "gtts_ready": True,
    }

@router.get("/health/ready")
async def health_ready(services = Depends(get_services)):
    """
    Deeper readiness check that exercises each component.
    """
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": check_asr_ready(services.asr),
        "llm_ready": check_llm_ready(services.llm),
        "gtts_ready": check_gtts_ready(),
    }
