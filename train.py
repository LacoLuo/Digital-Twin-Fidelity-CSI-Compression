import os 
import argparse
from scipy.io import savemat

from process import train_process
from config import configurations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train the model.")
    parser.add_argument(
            "-s", "--store_model_path", required=True, type=str,
            help="path of checkpoint")
    parser.add_argument(
            "-l", "--load_model_path", default=None, type=str,
            help="path of pretrained model")
    args = parser.parse_args()

    config = configurations()
    config.store_model_path = args.store_model_path
    config.load_model_path = args.load_model_path
    print('config:\n', vars(config))

    val_nmse = train_process(config)
    
    savemat(
        os.path.join(config.store_model_path, "val_nmse.mat"),
        {"val_nmse": val_nmse}
    )