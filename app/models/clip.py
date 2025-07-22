from torch import nn
import torch
import open_clip

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        embed_dim = clip_model.visual.output_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images, enable_grad=False):
        if enable_grad:
            # Allow gradients for Grad-CAM
            features = self.clip.encode_image(images)
        else:
            # Normal inference without gradients
            with torch.no_grad():
                features = self.clip.encode_image(images)
        return self.classifier(features)
    
clip_model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)

def Get_clip():
    clip = CLIPClassifier(clip_model, num_classes=5)
    return clip
