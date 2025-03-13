import os

import dotenv
import torch

dotenv.load_dotenv()

HF_REPO = os.getenv("HF_REPO")
CHECKPOINT_DIR = os.getenv("CHECKPOINTS_DIR")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
