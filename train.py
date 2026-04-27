import sys
# from time import time
import numpy as np

# import random as rd
import os

from pathlib import Path
import argparse
import json
# sys.path.append(str(Path(__file__).parent.parent))

# import setup_datasets.CIFAR10 as CIFAR
from network import neuralNet as NN
import paths

def log_params(net: NN.NeuralNet, param_folder_path: Path):
    
    print("\nLogging model settings...")
    
    weights_path = param_folder_path / paths.WEIGHTS_FILE
    biases_path = param_folder_path / paths.BIASES_FILE
    loss_and_activations_path = param_folder_path / paths.LOSS_AND_ACTIVATIONS_FILE
    train_mean_path = param_folder_path / paths.TRAIN_MEAN_FILE
    train_std_path = param_folder_path / paths.TRAIN_STD_FILE
    
    
    np.savez(
        weights_path,
        **{str(i): l.weights
        for i, l in enumerate(net.trainable_layers, 1)}
    )
    np.savez(biases_path,
        **{str(i): l.biases
        for i, l in enumerate(net.trainable_layers, 1)}
    )
    with open(loss_and_activations_path, "w") as f:
        
        f.write(net.loss_type)
        f.write('\n\n')
        for layer in net.trainable_layers:
            f.write(layer.activation_type)
            f.write('\n')
        
        # f.write('\n'+default_folder_name)
    
    np.save(train_mean_path, net.train_mean)
    np.save(train_std_path, net.train_std)
    
    if net.do_batch_norm:
        
        BN_folder_path = param_folder_path / paths.BN_FOLDER_NAME
        BN_moving_mean_path = BN_folder_path / paths.BN_MOVING_MEAN_FILE
        BN_moving_std_path = BN_folder_path / paths.BN_MOVING_STD_FILE
        BN_gamma_path = BN_folder_path / paths.BN_GAMMA_FILE
        BN_beta_path = BN_folder_path / paths.BN_BETA_FILE
        
        os.makedirs(BN_folder_path, exist_ok=True)

        
        np.savez(BN_moving_mean_path,
        **{str(i): l.BN_moving_avg_mean
        for i, l in enumerate(net.hidden_layers, 1)}
        )
        np.savez(BN_moving_std_path,
        **{str(i): l.BN_moving_avg_std
        for i, l in enumerate(net.hidden_layers, 1)}
        )
        np.savez(BN_gamma_path,
        **{str(i): l.BN_gamma
        for i, l in enumerate(net.hidden_layers, 1)}
        )
        np.savez(BN_beta_path,
        **{str(i): l.BN_beta
        for i, l in enumerate(net.hidden_layers, 1)}
        )
        
    print(f"Logging to {param_folder_path.name} finished!\n")


print("Setting up model...")

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
args = parser.parse_args()

config_file_name = args.config

assert config_file_name is not None, "User must specify --config arg (config file name in configs/)"

if ".json" not in config_file_name:
    config_file_name += ".json"
    
config_file_path = paths.TRAINING_CONFIGS_DIR / config_file_name
# config_file_path = "training_configs/"+config_file_name
# if ".json" not in config_file_path:
#     config_file_path += ".json"
    
with open(config_file_path) as f:
    cfg = json.load(f)

dataset = cfg["dataset"]
model = cfg["model"]
training_params_json = cfg["training"]

loss_type = model["loss"]

layer_params = model["layers"]
hidden_layers = layer_params["hidden"]
output_layer = layer_params["output"]

do_batch_norm = bool(model["do_batch_norm"])

do_log = bool(training_params_json["log_params"])
log_file_name = training_params_json["log_file_name"]


if dataset == "MNIST":
    import setup_datasets.MNIST as DATA
elif dataset == "FASHION_MNIST":
    import setup_datasets.FASHION_MNIST as DATA
elif dataset == "CIFAR10":
    import setup_datasets.CIFAR10 as DATA
else:
    assert False

x_train = DATA.x_train
y_train = DATA.y_train
y_train_one_hot = DATA.y_train_one_hot

x_test = DATA.x_test
y_test = DATA.y_test
y_test_one_hot = DATA.y_test_one_hot


input_dims = x_train.shape[1]

layers = [((-1, input_dims), "")]

for i in range(len(hidden_layers["dims"])):
    layers.append(((layers[-1][0][1], hidden_layers["dims"][i]), hidden_layers["activations"][i]))

layers.append(((layers[-1][0][1], output_layer["dim"]), output_layer["activation"]))
    
# layers = [input_dims,
#         #   ((784, 10), "relu"),
#           ((input_dims, 1024), "leakyrelu"),
#         #   ((2048, 1024), "leakyrelu"),  
#           ((1024, 512), "leakyrelu"),
#         #   ((784*2, 512), "leakyrelu"),
#         #   ((784, 128), "relu"),
#           ((512, 256), "leakyrelu"),
#           ((256, 128), "leakyrelu"),
#         #   ((784, 784*2), "relu"),
        
#         #   ((128, 64), "relu"),
        
#           ((128, 10), "softmax")
#           ]

net_params = {"loss": loss_type,
              "layers": layers,
              "batch_norm": do_batch_norm
              }


training_params = {"train_data": [x_train, y_train, y_train_one_hot],
               "test_data": [x_test, y_test, y_test_one_hot],
            #    "training_samples": len(x_train),
               "tests_per_epoch": training_params_json["tests_per_epoch"],
                "epochs": training_params_json["epochs"],
                "LR": training_params_json["LR"],
                "momentum": training_params_json["momentum"],
                "batch_size": training_params_json["batch_size"],
                "min_accuracy": training_params_json["min_accuracy_cutoff"],
                "test": bool(training_params_json["show_testing_accuracy"]),
                "%": bool(training_params_json["show_epoch_completion_%"]),
                "visual": bool(training_params_json["show_visual_loss"])
                }

if training_params_json["num_training_samples"] != -1:
    training_params["training_samples"] = training_params_json["num_training_samples"]

net = NN.NeuralNet(net_params)

if do_log:
    
    assert log_file_name != "", "Logging file name cannot be empty!"
    # root = Path(__file__).parent
    # # root = here.parent

    # # print(root)
    # params_path = root / "parameters"

    params_path = paths.PARAMETERS_DIR/f"{dataset}"

    # folder_name = ""
    # while folder_name == "":
    #     folder_name = input("Enter custom logging folder name: ")
    
    
    folder_logging_path = params_path / log_file_name
    if os.path.isdir(folder_logging_path):
        foo = input(f"Logging folder {log_file_name} already exists. Data will be overwritten. Continue? (Y/N): ")
        if "y" not in foo.casefold():
            sys.exit()
        
        
    # print(folder_logging_path)
    os.makedirs(folder_logging_path, exist_ok = True)
    # assert False

input("Start training: ")

try:
    net.train(training_params)
except KeyboardInterrupt:
    if do_log:
        import testing.metrics as M
        print("\n\n")
        M.evaluate(net, "MNIST")
        foo = input("You interrupted the program! Would you still like to log the parameters? (Y/N)")
        if "y" not in foo.casefold():
            do_log = False
            
input("continue: ")

if do_log:
    folder_logging_path = locals()["folder_logging_path"] #because of possibly unbound errors otherwise
    # folder_name = locals()["folder_name"]
    
    log_params(net, folder_logging_path)
    

input("Press enter to exit.")
