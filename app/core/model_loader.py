import torch
import logging
import torch.nn as nn
from app.models.friendnet import FriendNet
from app.models.resnet import Get_resnet
from app.models.clip import Get_clip
from app.config import DEVICE

logger = logging.getLogger(__name__)

async def load_models(models_dict: dict):
    try:
        logger.info("Loading custom model...")
        model = FriendNet().to(DEVICE)
        model.load_state_dict(torch.load("friendnet.pth", map_location=DEVICE))
        model.eval()
        models_dict["friendnet"] = model
        logger.info("Friendnet loaded")

        logger.info("Loading ResNet model...")
        resnet = Get_resnet().to(DEVICE)
        resnet.load_state_dict(torch.load("resnet.pth", map_location=DEVICE))
        resnet.eval()
        models_dict["resnet"] = resnet
        logger.info("ResNet loaded")

        logger.info("Loading Clip model...")
        clip = Get_clip().to(DEVICE)
        clip.load_state_dict(torch.load("clip.pth", map_location=DEVICE))
        clip.eval()
        models_dict["clip"] = clip
        logger.info("Clip loaded")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
