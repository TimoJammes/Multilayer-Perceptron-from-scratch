from pathlib import Path

ROOT = Path(__file__).parent

PARAMETERS_DIR = ROOT / "parameters"
TRAINING_CONFIGS_DIR = ROOT / "training_configs"

BN_FOLDER_NAME = "batch_norm"

WEIGHTS_FILE = "weights.npz"
BIASES_FILE = "biases.npz"
LOSS_AND_ACTIVATIONS_FILE = "loss_and_activations.txt"
TRAIN_MEAN_FILE = "train_mean.npy"
TRAIN_STD_FILE = "train_std.npy"
BN_MOVING_MEAN_FILE = "BN_moving_mean.npz"
BN_MOVING_STD_FILE = "BN_moving_std.npz"
BN_GAMMA_FILE = "BN_gamma.npz"
BN_BETA_FILE = "BN_beta.npz"