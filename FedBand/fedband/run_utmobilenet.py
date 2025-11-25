from fedband.config import set_seed, SEED
from fedband.training import train_utmobilenet

if __name__ == "__main__":
    set_seed(SEED)
    # TODO: replace with your actual path to the Wild Test Data folder
    base_path = "/path/to/Wild_Test_Data"
    train_utmobilenet(base_path)

