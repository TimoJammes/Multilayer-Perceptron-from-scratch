# Neural Network from Scratch

A fully custom feedforward neural network built from scratch using NumPy. Tested on image classification with standard benchmarks, with support for batch normalization, SGD with momentum, interactive visualization, and JSON-based configuration.

---

## Features

- Dense (fully-connected) network with configurable depth and width
- Batch normalization (optional, per layer)
- SGD optimizer with momentum
- Z-score input normalization using training set statistics
- Mini-batch training with shuffling each epoch
- Early stopping via minimum accuracy threshold
- Real-time loss/accuracy visualization during training
- Interactive test-set browser to inspect predictions
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- Full model persistence — save and reload trained weights, biases, and normalization stats

---

## Supported Datasets

| Dataset | Images | Input Size | Classes | Results
|---|---|---|---|---|
| MNIST | 60k train / 10k test | 28×28 grayscale (784) | 10 digits | ~98% Accuracy
| Fashion MNIST | 60k train / 10k test | 28×28 grayscale (784) | 10 clothing categories | ~90% Accuracy
| CIFAR-10 | 50k train / 10k test | 32×32 RGB (3072) | 10 object categories | ~50% Accuracy

Datasets are loaded via `tensorflow.keras.datasets` and normalized to `[0, 1]` before training.

---

## Project Structure

```
neuralNetwork/
├── network/
│   ├── neuralNet.py          # Core NeuralNet class and training loop
│   ├── layers.py             # Layer definitions (Input, Trainable)
│   └── trainingVisualizer.py # Real-time PyGame loss/accuracy plot
│
├── setup_datasets/ #dataset imports and pre-processing
│   ├── MNIST.py
│   ├── FASHION_MNIST.py
│   └── CIFAR10.py
│
├── testing/
│   ├── metrics.py            # Accuracy, F1, confusion matrix...
│   ├── visual.py             # Interactive prediction viewer
|   ├── MNIST_user_input.py   # Draw-and-predict digit recognition demo
│   └── import_params.py      # Load a saved model from parameters/
│
├── visualize/
│   └── browse_dataset.py     # Browse raw dataset images interactively
│
├── training_configs/
│   ├── config.json # Training configuration (edit this) 
|   └── ...         
│
├── parameters/               # Saved model weights (auto-created)
├── train.py                  # Training entry point
├── paths.py                  # Path constants
└── requirements.txt
```

---

## Getting Started

Setting up a virtual environment for the project is recommended. The project was built and tested on python 3.12.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Understanding args

For any file that uses arguments in command line, you may type ```python -m "file" --help``` to get more information.

### Draw-and-Predict digit recognition demo

```bash
python -m testing.MNIST_user_input (--params_folder <model_name>)
```

### Train a model

```bash
python train.py --config config
```

Pass any config filename from `training_configs/`.

### Evaluate a trained model

```bash
python -m testing.metrics --dataset MNIST (--params_folder <model_name>)
```

### Browse predictions interactively

```bash
python -m testing.visual --dataset MNIST --params_folder <model_name> --find_incorrect 0 --randomize_order 1
```

- Arrow keys to navigate samples
- Toggle `--find_incorrect 1` to auto-skip to misclassified examples

### Browse the raw dataset

```bash
python -m visualize.browse_dataset --dataset MNIST --randomize 1
```

---

## Configuration

All training options live in a JSON config file under `training_configs/`. The default is `config.json`.

```json
{
  "dataset": "MNIST",
  "model": {
    "loss": "crossentropy",
    "layers": {
      "hidden": {
        "dims": [512, 256, 128, 64],
        "activations": ["leakyrelu", "leakyrelu", "leakyrelu", "leakyrelu"]
      },
      "output": {
        "dim": 10,
        "activation": "softmax"
      }
    },
    "do_batch_norm": "True"
  },
  "training": {
    "epochs": 50,
    "LR": 0.01,
    "momentum": 0.9,
    "batch_size": 256,
    "min_accuracy_cutoff": 98,
    "tests_per_epoch": 1024,
    "num_training_samples": -1,
    "show_testing_accuracy": 1,
    "show_epoch_completion_%": 1,
    "show_visual_loss": 0,
    "log_params": 0,
    "log_file_name": "my_model"
  }
}
```

### Key Parameters

| Parameter | Description |
|---|---|
| `dataset` | `MNIST`, `FASHION_MNIST`, or `CIFAR10` |
| `dims` | Width of each hidden layer (list) |
| `activations` | Per-layer activation: `relu`, `leakyrelu`, `sigmoid` |
| `do_batch_norm` | Enable batch normalization for all hidden layers |
| `LR` | Learning rate |
| `momentum` | Momentum coefficient (0 = plain SGD, 0.9 = standard) |
| `batch_size` | Mini-batch size |
| `epochs` | Number of training epochs |
| `min_accuracy_cutoff` | Stop early if test accuracy reaches this threshold (%) |
| `num_training_samples` | Limit training data size (`-1` = use full dataset) |
| `log_params` | Save model to disk after training |
| `log_file_name` | Folder name under `parameters/<dataset>/` for saved model |
| `show_visual_loss` | Launch real-time loss plot during training (PyGame) |

---

## Training Details

### Optimizer — SGD with Momentum

Weights are updated using accumulated gradient velocity:

```
velocity = momentum * velocity + gradient
weights  = weights - LR * velocity
```

Setting `momentum = 0` reduces this to standard SGD.

### Batch Normalization

When enabled, each hidden layer applies batch normalization before its activation:

```
y = γ * (x - μ) / √(σ² + ε) + β
```

- `γ` and `β` are learned per-neuron parameters
- Moving averages of mean and variance are tracked for inference
- Gradients are computed through the normalization for end-to-end training

### Input Normalization

Training set mean and standard deviation are computed once and applied to all splits:

```
x_normalized = (x - train_mean) / (train_std + ε)
```

These statistics are saved alongside the model so inference uses the same normalization.

### Weight Initialization

Hidden layer weights are initialized using He initialization (uniform distribution scaled by `√(2 / fan_in)`), suited for ReLU-family activations.

---

## Model Persistence

When `log_params` is enabled, the trained model is saved to `parameters/<dataset>/<log_file_name>/`:

```
parameters/
└── MNIST/
    └── my_model/
        ├── weights.npz
        ├── biases.npz
        ├── loss_and_activations.txt
        ├── train_mean.npy
        ├── train_std.npy
        └── batch_norm/
            ├── BN_moving_mean.npz
            ├── BN_moving_std.npz
            ├── BN_gamma.npz
            └── BN_beta.npz
```

---

## Evaluation Metrics

Running `testing/metrics.py` on a saved model reports:

- Overall accuracy
- Per-class accuracy
- Macro precision, recall, and F1 score
- Full confusion matrix

---

## Dependencies

```
numpy
Pillow
pygame
scipy
tensorflow  (dataset loading only)
```

---

## Author
Timothé Jammes — [GitHub profile link] — timothe.jammes@gmail.com