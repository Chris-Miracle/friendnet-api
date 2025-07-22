from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.model_loader import load_models
from app.routes import predict, misc
from app.config import DEVICE, models_dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# models_dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models(models_dict)
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="FriendNet AI API",
    version="1.0.0",
    description="Advanced neural network API for friend recognition",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/predict")
app.include_router(misc.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
