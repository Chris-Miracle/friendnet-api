from torchvision import transforms
import open_clip


def get_advanced_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Add second normalization to match training
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_clip_transforms():
    _, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    return clip_preprocess