from fastapi import APIRouter
from datetime import datetime
from app.config import models_dict, DEVICE, FRIENDS

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models_dict),
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/models")
async def get_available_models():
    return {
        "available_models": list(models_dict.keys()),
        "friends": FRIENDS,
        "device": str(DEVICE)
    }
