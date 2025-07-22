import torch.nn.functional as F
import numpy as np
import torch

from app.config import FRIENDS

async def get_model_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        max_prob = float(np.max(probs))
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        confidence = max_prob * (1 - entropy / np.log(len(FRIENDS)))

        return {
            "probabilities": {FRIENDS[i]: float(probs[i]) for i in range(len(FRIENDS))},
            "predicted_class": FRIENDS[np.argmax(probs)],
            "confidence": confidence,
            "max_probability": max_prob,
            "entropy": entropy
        }
