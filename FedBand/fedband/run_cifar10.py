from fedband.config import set_seed, SEED
from fedband.training import train_cifar10

if __name__ == "__main__":
    set_seed(SEED)
    train_cifar10()

