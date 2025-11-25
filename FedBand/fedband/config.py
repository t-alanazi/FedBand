import random
import numpy as np
import torch

SEED = 0
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
NUM_CLIENTS = 100
BANDWIDTH_LIMIT = 10      # β
LOCAL_EPOCHS = 3
NUM_ROUNDS = 1000
INITIAL_ROUNDS = 1
DIRICHLET_ALPHA = 0.5
ALLOCATION_METRIC = "val_loss"  # or "grad_norm"
VALUE_BYTES = 4
INDEX_BYTES = 4
MIN_NNZ = 1


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


