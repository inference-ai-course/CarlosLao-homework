"""
routers/memory.py
-----------------
FastAPI router for memory inspection and reset.
"""

from fastapi import APIRouter, Depends
from ..memory import MemoryService

router = APIRouter()

def get_services(request):
    return request.app.state.services

@router.get("/memory")
async def get_memory(services = Depends(get_services)):
    """
    Retrieve the current in-memory conversation history.
    """
    return await services.memory.get_recent()

@router.delete("/memory")
async def clear_memory(services = Depends(get_services)):
    """
    Clear the in-memory conversation history and delete transcript file.
    """
    await services.memory.clear()
    return {"cleared": True}
