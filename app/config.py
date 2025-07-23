import torch

FRIENDS = ["jacob", "luke", "osama", "tatyana", "uzair"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dict = {}  # Shared globally
