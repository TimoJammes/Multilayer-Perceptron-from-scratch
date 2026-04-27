import os
from pathlib import Path
import numpy as np

import network.neuralNet as NN
import paths

def NN_from_params(dataset: str, params_folder_name: str):

    # here = Path(__file__).parent
    # root = here.parent

    # params_path = root / "parameters"
    params_path = paths.PARAMETERS_DIR / f"{dataset}" / params_folder_name

    weights = np.load(params_path / paths.WEIGHTS_FILE)
    biases = np.load(params_path / paths.BIASES_FILE)

    with open(params_path / paths.LOSS_AND_ACTIVATIONS_FILE, "r") as f:
        words = f.read().split()

    loss_type = words[0]
    activations = words[1:]

    layers: list[object] = [((-1, len(weights["1"][0])), "")]

    for layer in weights:
        if layer == "1":
            layers.append(((len(weights["1"][0]), len(weights[layer])), activations[int(layer)-1]))
            continue
        layers.append(((len(weights[str(int(layer)-1)]), len(weights[layer])), activations[int(layer)-1]))

    BN_path = params_path / paths.BN_FOLDER_NAME
    
    do_batch_norm = os.path.exists(BN_path)

    params = {"loss": loss_type,
              "layers": layers,
              "batch_norm": do_batch_norm
              }

    net = NN.NeuralNet(params)

    if do_batch_norm:
        
        BN_moving_avg_mean = np.load(BN_path / paths.BN_MOVING_MEAN_FILE)
        BN_moving_avg_std = np.load(BN_path / paths.BN_MOVING_STD_FILE)
        BN_gamma = np.load(BN_path / paths.BN_GAMMA_FILE)
        BN_beta = np.load(BN_path / paths.BN_BETA_FILE)

        for l in net.hidden_layers:
            l.BN_moving_avg_mean = BN_moving_avg_mean[str(l.depth)]
            l.BN_moving_avg_std = BN_moving_avg_std[str(l.depth)]
            l.BN_beta = BN_beta[str(l.depth)]
            l.BN_gamma = BN_gamma[str(l.depth)]

    
    if Path(params_path/paths.TRAIN_MEAN_FILE).exists():
        net.train_mean = np.load(params_path / paths.TRAIN_MEAN_FILE)
        net.train_std = np.load(params_path / paths.TRAIN_STD_FILE)

        net.is_set_normalize = True

    for layer in weights:
        net.trainable_layers[int(layer)-1].weights = weights[layer]
        net.trainable_layers[int(layer)-1].biases = biases[layer]

    return net
