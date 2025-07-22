import torch

FRIENDS = ["tatyana", "uzair", "jacob", "luke", "osama"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dict = {}  # Shared globally
