from fastapi import APIRouter, UploadFile, File, HTTPException, Form
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
async def predict_image(file: UploadFile = File(...), models: str = Form(None), enhance_image: bool = Form(False)):
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # DEBUG: Check image properties
    print(f"DEBUG - Original image size: {image.size}")
    print(f"DEBUG - Image mode: {image.mode}")
    enhance_image = False

    if enhance_image:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)

    # Parse model list and validate
    if models is None:
        raise HTTPException(status_code=400, detail="models parameter is required")
    
    if models == "all":
        model_list = list(models_dict.keys())
    else:
        # Split by comma and strip whitespace
        model_list = [m.strip() for m in models.split(",")]
        
        # Validate that all requested models exist
        invalid_models = [m for m in model_list if m not in models_dict]
        if invalid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model(s): {invalid_models}. Available models: {list(models_dict.keys())}"
            )

    # DEBUG: Add logging to see what's happening
    print(f"DEBUG - Input models parameter: '{models}'")
    print(f"DEBUG - Parsed model_list: {model_list}")
    print(f"DEBUG - Available models_dict.keys(): {list(models_dict.keys())}")

    predictions = {}

    # Process each model
    for model_name in model_list:
        print(f"DEBUG - Processing model: {model_name}")
        
        # Use CLIP-specific transform if applicable
        if model_name.lower() == "clip":
            transform = get_clip_transforms()
        else:
            transform = get_advanced_transforms()

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # DEBUG: Check tensor properties
        print(f"DEBUG - Image tensor shape: {image_tensor.shape}")
        print(f"DEBUG - Image tensor dtype: {image_tensor.dtype}")
        print(f"DEBUG - Image tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
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
        print(f"DEBUG - Added prediction for {model_name}: {pred['predicted_class']} (confidence: {pred['confidence']:.3f})")

    print(f"DEBUG - Final predictions keys: {list(predictions.keys())}")
    print(f"DEBUG - len(model_list): {len(model_list)}, len(predictions): {len(predictions)}")

    # Only add ensemble if multiple models were requested AND processed
    if len(model_list) > 1 and len(predictions) > 1:
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