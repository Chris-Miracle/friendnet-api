import os
import torch
import gdown
import logging
import torch.nn as nn

from app.models.friendnet import FriendNet
from app.models.resnet import Get_resnet
from app.models.clip import Get_clip
from app.config import DEVICE

logger = logging.getLogger(__name__)

def download_model_if_missing(path, gdrive_url):
    if not os.path.exists(path):
        print(f"Downloading {path} from Google Drive...")
        gdown.download(gdrive_url, path, quiet=False)
    else:
        print(f"{path} already exists, skipping download.")

async def load_models(models_dict: dict):
    try:
        # Paths to local files
        friendnet_path = "friendnet.pth"
        resnet_path = "resnet.pth"
        clip_path = "clip.pth"

        # GDrive URLs
        download_model_if_missing(friendnet_path, "https://drive.google.com/uc?id=1UP-hCjieshxPHJKv4uITMwWpjf2tbzb4")
        download_model_if_missing(resnet_path, "https://drive.google.com/uc?id=1MSgeQyIa9Y1wvT-OTF5kKUIgOIUQjz3I")
        download_model_if_missing(clip_path, "https://drive.google.com/uc?id=1jWFa-apGQmTfH0UeSjM6OYYlAz_IGYuF")

        # Load FriendNet
        logger.info("Loading FriendNet model...")
        model = FriendNet().to(DEVICE)
        model.load_state_dict(torch.load(friendnet_path, map_location=DEVICE))
        model.eval()
        models_dict["friendnet"] = model
        logger.info("FriendNet loaded ✅")

        # Load ResNet
        logger.info("Loading ResNet model...")
        resnet = Get_resnet().to(DEVICE)
        resnet.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
        resnet.eval()
        models_dict["resnet"] = resnet
        logger.info("ResNet loaded ✅")

        # Load CLIP
        logger.info("Loading CLIP model...")
        clip = Get_clip().to(DEVICE)
        clip.load_state_dict(torch.load(clip_path, map_location=DEVICE))
        clip.eval()
        models_dict["clip"] = clip
        logger.info("CLIP loaded ✅")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise
