from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance
import io
import time
import numpy as np
from app.core.transforms import get_advanced_transforms, get_clip_transforms
from app.core.predictor import get_model_prediction
from app.config import DEVICE, FRIENDS, models_dict
from datetime import datetime

router = APIRouter()

@router.post("")
async def predict_image(file: UploadFile = File(...), models: str = "all", enhance_image: bool = False):
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    enhance_image = False

    if enhance_image:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)

    model_list = models.split(",") if models != "all" else list(models_dict.keys())
    predictions = {}

    for model_name in model_list:
        if model_name in models_dict:
            # Use CLIP-specific transform if applicable
            if model_name.lower() == "clip":
                transform = get_clip_transforms()
            else:
                transform = get_advanced_transforms()

            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            pred = await get_model_prediction(models_dict[model_name], image_tensor)

            # Convert all numpy or tensor floats to native python floats
            cleaned_pred = {
                "predicted_class": pred["predicted_class"],
                "confidence": float(pred["confidence"]),
                "max_probability": float(pred.get("max_probability", 0.0)),
                "entropy": float(pred.get("entropy", 0.0)),
                "probabilities": {k: float(v) for k, v in pred["probabilities"].items()}
            }

            predictions[model_name] = cleaned_pred

    if len(predictions) > 1:
        ensemble_probs = np.mean(
            [list(p["probabilities"].values()) for p in predictions.values()],
            axis=0
        )
        predictions["ensemble"] = {
            "probabilities": {FRIENDS[i]: float(ensemble_probs[i]) for i in range(len(FRIENDS))},
            "predicted_class": FRIENDS[np.argmax(ensemble_probs)],
            "confidence": float(np.max(ensemble_probs)),
            "max_probability": float(np.max(ensemble_probs)),
            "entropy": float(-np.sum(ensemble_probs * np.log(ensemble_probs + 1e-8)))
        }

    return JSONResponse({
        "predictions": predictions,
        "metadata": {
            "processing_time": round(time.time() - start_time, 3),
            "image_size": image.size,
            "models_used": model_list,
            "device": str(DEVICE),
            "timestamp": datetime.now().isoformat()
        }
    })
