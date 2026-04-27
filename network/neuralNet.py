import random as rd
# from math import exp
import numpy as np
import time
from collections.abc import Iterable
# import pygame
# import logging
import sys
import numbers

from . import layers

def is_number(x):
    return isinstance(x, numbers.Number) or isinstance(x, np.generic)

def is_iterable(l):
    return isinstance(l, Iterable) and not isinstance(l, (str, bytes))

def dist(min, max):
    return rd.uniform(min, max)



EPS = 1e-8


class TrainingParams:
    

    def check_training_params(self, params):
        
        for p in params:
            assert p in self.valid_params, "Invalid training parameter!"

        for p in self.required_params:
            assert p in params, "training params must include required params!"
        # assert "train_data" in params, "training params must include train_data (training data)"
        # assert "epochs" in params, "training params must include epochs"
        # assert "LR" in params, "training params must include LR (learning rate)"
        
        # print(len(params["train_data"]))
        # assert all([isIterable(x) and isIterable(y) and len(x)==len(y) for x, y in params["train_data"]])
        
        # if "test_data" in params:
        #     assert all([isIterable(x) and isIterable(y) and len(x)==len(y) for x, y in params["test_data"]])
        
        if "%" in params:
            assert isinstance(params["%"], bool)
        if "visual" in params:
            assert isinstance(params["visual"], bool)
        
        if "test" in params:
            assert isinstance(params["test"], bool)
            
        if "batch_size" in params:
            assert isinstance(params["batch_size"], int)
            assert 0 < params["batch_size"] <= NeuralNet.max_batch_size, "Specified batch_size must be between 1 and max_batch_size!"
        
        if "min_accuracy" in params:
            assert isinstance(params["min_accuracy"], float) or isinstance(params["min_accuracy"], int)
            assert 0 <= params["min_accuracy"] <= 100
        
        # if "minCompletion" in params:
        #     assert isinstance(params["minCompletion"], float) or isinstance(params["min_accuracy"], int)
        #     assert 0 <= params["minCompletion"] <= 100
        
        if "momentum" in params:
            assert isinstance(params["momentum"], float) or isinstance(params["momentum"], int)
            assert 0 <= params["momentum"] <= 1
        
        # if "tests_per_epoch" in params:
        #     assert 0 <= params["tests_per_epoch"] <= len(params[""]), f"tests_per_epoch: 0-{NeuralNet.max_tests_per_epoch}!"
  
    def __init__(self, params):
        
        self.valid_params = NeuralNet.all_train_params
        self.required_params = NeuralNet.required_train_params
        
        self.check_training_params(params)
        
        x_train, y_train, y_train_one_hot = params["train_data"]
        x_test, y_test, y_test_one_hot = params["test_data"] if "test_data" in params else (None, None, None)

        epochs = params["epochs"]
        learning_rate = params["LR"]
        
        momentum_coeff = params["momentum"] if "momentum" in params else 0
        minibatch_size = params["batch_size"] if "batch_size" in params else NeuralNet.default_batch_size
        
        min_accuracy_percent = params["min_accuracy"] if "min_accuracy" in params else NeuralNet.default_min_accuracy
        # minCompletion = params["minCompletion"] if "minCompletion" in params else NeuralNet.default_min_completion
        
        num_samples = params["training_samples"] if "training_samples" in params else len(x_train)
        tests_per_epoch = min(len(x_test), params["tests_per_epoch"]) if "tests_per_epoch" in params else min(NeuralNet.default_test_examples_tests_per_epoch, len(x_test))#type: ignore

        do_percentage = params["%"] if "%" in params else False
        do_visual = params["visual"] if "visual" in params else False
        do_test = params["test"] if "test" in params  else False
        
        # net.setNormalize(x_train)
        indices = np.random.choice(len(x_train), size=num_samples, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]
        y_train_one_hot = y_train_one_hot[indices]
        # training = list(zip(x_train[indices], y_train[indices], y_train_one_hot[indices]))
        # training = [TrainingExample(*t) for t in training]
        x_train_accuracy_calc = x_train[:NeuralNet.num_training_example_test_per_epoch]
        y_train_accuracy_calc = y_train[:NeuralNet.num_training_example_test_per_epoch]
        y_train_one_hot_accuracy_calc = y_train_one_hot[:NeuralNet.num_training_example_test_per_epoch]
        # training_accuracy_calc_one_hot_correct_indices = [t.target_readable for t in training_for_accuracy_rate]
        
        if x_test is None:
            do_test = False
        if do_test:
            indices = np.random.choice(len(x_test), size=tests_per_epoch, replace=False)#type: ignore
            x_test = x_test[indices]#type: ignore
            y_test = y_test[indices]#type: ignore
            y_test_one_hot = y_test_one_hot[indices]#type: ignore                    
        
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_one_hot = y_train_one_hot
        self.x_train_accuracy_calc = x_train_accuracy_calc
        self.y_train_accuracy_calc = y_train_accuracy_calc
        self.y_train_one_hot_accuracy_calc = y_train_one_hot_accuracy_calc
        
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_one_hot = y_test_one_hot
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.momentum_coeff = momentum_coeff
        self.minibatch_size = minibatch_size
        
        self.tests_per_epoch = tests_per_epoch
        
        self.min_accuracy_percent = min_accuracy_percent
        # self.minCompletion = minCompletion
        
        self.do_percentage = do_percentage
        self.do_visual = do_visual
        self.do_test = do_test
        
        
        
class NeuralNet:
    
    activation_types = ["relu", "leakyrelu","sigmoid", "softmax"]
    loss_types = ["crossentropy", "mse"]

    net_params = {"loss", "layers", "batch_norm"}
     
    def check_net_params(self, params):
        
        for p in params:
            assert p in self.net_params, p+" is not a valid network parameter!"
        # if "loss" not in params or "layers" not in params:
        assert "loss" in params and "layers" in params, "Network params must include loss type and layers!"
        
        assert params["loss"] in self.loss_types, "loss type not valid"
        l = params["layers"]
        
        assert len(l) >=2 
        
        for i in range(len(l)):
            # if i == 0:
            #     assert isinstance(l[i], int)
            #     continue
            assert isinstance(l[i], tuple)
            assert len(l[i]) == 2

            
            assert isinstance(l[i][0][0], int)
            assert isinstance(l[i][0][1], int)
            assert isinstance(l[i][1], str)
            assert len(l[i][0]) == 2
        
        assert isinstance(params["batch_norm"], bool)
    
        # if "momentum" in params:
        #     assert isinstance(params["momentum"], float)        

    def __init__(self, params):
        """_summary_

        Args:
            params (dict): all network parameters. See NeuralNet __init__ for more details.
        Required params: 
            -loss: "mse", "crossentropy"
            -layers: (num_inputs, ((num_inputs, layer1_width), layer1_activation), ..., (layer(n-1)_width, output_width))
        Optional params:
            
        """
        
        
        self.check_net_params(params)
        self.loss_type = params["loss"]
        
        layer_params = params["layers"]
        # print(layerParams)
        self.do_batch_norm = params["batch_norm"] if "batch_norm" in params else False
        
        num_layers = len(layer_params)
        
        self.input_layer = layers.InputLayer(layer_params[0][0][1])
        # net.inputLayer = layers.InputLayer(layerParams[0])
        
        self.layers: list[layers.Layer] = [self.input_layer]
        
        self.hidden_layers = []
        # print(layerParams)
        for i in range(1, num_layers-1):
            in_width, out_width = layer_params[i][0]
            # print(inWidth, outWidth)
            activation = layer_params[i][1]
            layer = layers.TrainableLayer(i, in_width, out_width, activation, self.do_batch_norm)
            self.layers.append(layer)
            self.hidden_layers.append(layer)
        
        if (self.loss_type == "crossentropy") != (layer_params[-1][1]=="softmax"):
            assert False, "crossentropy loss must be paired with softmax"
            
        # if net.lossType == "crossentropy":
        self.output_layer = layers.TrainableLayer(num_layers-1, layer_params[-1][0][0], layer_params[-1][0][1], layer_params[-1][1], do_batch_norm=False, is_output=True)
        # else:
        #     assert False, "setup MSE"
            
        self.layers.append(self.output_layer)        
        
        self.trainable_layers = self.hidden_layers + [self.output_layer]
        
        self.max_depth = num_layers-1        
        
        self.num_parameters = sum(self.layers[i].width * (self.layers[i-1].width + 1) 
            for i in range(1, len(self.layers)))

        self.is_set_normalize = False
    
    def set_normalize(self, X):
        
        self.train_mean = np.mean(X, axis=0)
        self.train_std = np.std(X, axis=0)
        
        self.is_set_normalize = True
        
    def normalize_data(self, X):
        
        if not self.is_set_normalize:
            raise PermissionError("Must set network normalization (train and std) before normalizing data!")
        
        if hasattr(self, "train_mean") and hasattr(self, "train_std"):
            return (X-self.train_mean) / (self.train_std+EPS)
        raise AttributeError("No train_mean/std stored in network!")
        # return X / 255
        # x_test = (x_test-net.train_mean) / train_std

    
    all_train_params = {"train_data", "epochs", "LR",
                "batch_size", "min_accuracy",
                "test_data", "%", "visual", "test",
                "momentum", "num_samples", "tests_per_epoch"}
    required_train_params = {"train_data", "epochs", "LR"}
    
    default_batch_size = 256
    max_batch_size = 512
    default_min_accuracy = 98
    default_min_completion = 10
    # max_tests_per_epoch = 4096
    default_test_examples_tests_per_epoch = 1024
    num_training_example_test_per_epoch = 1024
    
    
    def display_progression(self, curr_epoch, epochs, s, precision):
        est = (time.time()-s)*(epochs/curr_epoch-1)
        unit = "s"
        if est > 60:
            est /= 60
            unit = "m"
        if est > 60:
            est /= 60
            unit = "h"

        print(f"Current epoch: {curr_epoch+1} (/{epochs} max)")
        print(f"Estimated training time left: {round(est, 2)}{unit}.")

    def display_accuracy(self, train_avg, test_avg):        
        print(f"Training accuracy: {round(train_avg*100, 3)}%")
        print(f"Testing accuracy: {round(test_avg*100, 3)}%")

    def calc_avg_accuracy(self, x_train, y_train, x_test, y_test):
        test_net_output = self.inference_feedforward(x_test)
        pred = np.argmax(test_net_output, axis=0)
        test_avg = np.mean(pred == y_test)

        train_net_output = self.inference_feedforward(x_train)
        pred = np.argmax(train_net_output, axis=0)
        train_avg = np.mean(pred == y_train)
        
        return test_net_output, train_avg, test_avg

    def do_line_rewrites(self, percentage, test):
        if percentage:                                
            sys.stdout.write("\033[F\033[F")
            sys.stdout.write("\033[F")
        if test:
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
        if percentage or test:
            sys.stdout.flush()

    def train(self, params):
        """_summary_

        Args:
            params (dict): all network parameters. See NeuralNet train for more details.
        Required params: 
            -train_data: [train_inputs, train_outputs] (input and outputs are arrays of size (#inputs, input_layer) and (#inputs, output_layer) of NN)
            -epochs
            -LR: learning rate for gradient descent
        Optional params:
            -training_samples: number of train_data samples to train on (len(train_data) by default)
            -test_data: [train_inputs, train_outputs] examples to test on, up to tests_per_epoch per epoch.
            -tests_per_epoch: num tests from test_data per epoch (min(max_tests_per_epoch, #test_data) by default)
            -test: bool, (requires test_data to be given) test accuracy on test_data during training and stop training if min_accuracy reached (false by default)
            -batch_size: for training, 1-max_batch_size, (min(max_batch_size, #training) by default)
            -min_accuracy: minimum % of accuracy on test_data to stop training at (99 by default)
            -%: bool, show progression in console (false by default)
            -visual: bool, show loss progression in pygame (false by default)
            -momentum: coeff for gradient descent momentum (0 by default, no momentum)
        """
        
        import testing.metrics as M
        
        p = TrainingParams(params)
        
        self.set_normalize(p.x_train)
                
        p.x_train = self.normalize_data(p.x_train)

        self.sanity_check(x_train=p.x_train_accuracy_calc, y_train_one_hot=p.y_train_one_hot_accuracy_calc)

        if p.do_visual:
            from . import trainingVisualizer as TV
            TV.setup()
            TV.pygame.display.update()

        test_loss = None

        precision = max(0, round(np.log10(p.epochs))-2) #for progression display
        
        print("\nTraining started... (Ctrl+C to end early)")
        
        start_time = time.time()

        for curr_epoch in range(1, p.epochs+1):
                        
            if p.do_visual:
                weighted_epoch_loss_sum = 0
                    
            id_shuffle = np.random.permutation(p.num_samples)
            
            batch_count = 0
            
            for start_index in range(0, p.num_samples, p.minibatch_size):

                if p.do_visual:
                    TV.pygame.event.pump()
                
                batch_indices = id_shuffle[start_index:start_index+p.minibatch_size]
                batch_x_train = p.x_train[batch_indices]
                batch_y_train_one_hot = p.y_train_one_hot[batch_indices]
                
                curr_batch_net_output = self.training_feedforward(batch_x_train)
                self.backpropagate(p.learning_rate, p.momentum_coeff, batch_y_train_one_hot)

                if p.do_visual:
                    loss = self.compute_loss(batch_y_train_one_hot, curr_batch_net_output)                    
                    weighted_epoch_loss_sum += loss * len(batch_x_train)
                
                batch_count += 1
                if p.do_percentage:
                    if batch_count % 5 == 0:
                        print(f"Current epoch progress: {(start_index+p.minibatch_size)/p.num_samples*100:.2f}%", end='\r')
                    

            if p.do_percentage:
                print()
                self.display_progression(curr_epoch, p.epochs, start_time, precision)
            
            if p.do_test:
                #returns net_test_output to avoid re feedforwarding test for do_visual
                net_test_output, train_avg, test_avg = self.calc_avg_accuracy(p.x_train_accuracy_calc, p.y_train_accuracy_calc, p.x_test, p.y_test)
                self.display_accuracy(train_avg, test_avg)
                
                if test_avg >= p.min_accuracy_percent/100:
                    print('\n\n')
                    print(f"Testing accuracy exceeds {p.min_accuracy_percent}% and training % > 10, training stopped.")
                    print(f"Completed {curr_epoch} epochs out of {p.epochs}.")
                    print(f"Training finished! Took {round(time.time()-start_time, 2)}s!\n")
                    return
            
            self.do_line_rewrites(p.do_percentage, p.do_test)
            # M.evaluate(self, "MNIST")
            
            if p.do_visual:
                avg_loss = weighted_epoch_loss_sum/p.num_samples
                if p.do_test:
                    test_loss = self.compute_loss(p.y_test_one_hot, net_test_output)
                TV.update_screen(avg_loss, test_loss, p.epochs, curr_epoch, train_avg, test_avg)                    
                            
        print('\n\n\n')
        print(f"Training finished! Took {round(time.time()-start_time, 2)}s!\n")

    # add visualizer
    
    def inference_feedforward(self, x):
        
        # print(x.shape)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        #     x
        assert x.shape[1]==self.input_layer.width
        # assert False
        
        # if self.do_batch_norm:
        if hasattr(self, "train_mean"):
            x = self.normalize_data(x)
        
        self.input_layer.feed(x.T)
        
        prev_layer_output = self.input_layer.output
        
        for layer in self.trainable_layers:
            # if layer.depth == 0:
            #     continue
            
            layer.inference_feed(prev_layer_output)
            
            #for next loop
            prev_layer_output = layer.output
            
        return prev_layer_output
        
        
    def training_feedforward(self, x_train):
        
        self.input_layer.feed(x_train.T)
        prev_layer_output = self.input_layer.output
        
        for layer in self.trainable_layers:
            
            layer:layers.TrainableLayer = layer
            layer.training_feed(prev_layer_output)
            
            #for next loop
            prev_layer_output = layer.output
            
        return prev_layer_output
            
    def backpropagate(self, LR, momentum_coeff, y_train_target_one_hot):

        if self.output_layer.activation_type != "softmax":
            assert False
        
        num_examples_curr_batch = len(y_train_target_one_hot)
        
        if self.output_layer.activation_type == "softmax" and self.loss_type == "crossentropy":
            self.output_layer.grad_E_wrt_Z = 1/num_examples_curr_batch * (self.output_layer.activation - np.column_stack(y_train_target_one_hot))
            #1/n: average over all examples
        
        elif self.loss_type == "mse" and self.output_layer.activation_type != "softmax":
            d_activation = self.output_layer.d_activation_function
            
            self.output_layer.grad_E_wrt_Z = 1/num_examples_curr_batch * (self.output_layer.activation - np.column_stack(y_train_target_one_hot)) * d_activation(self.output_layer.z)
            #1/n: average over all examples
        else:
            assert False
             
        for depth in range(self.max_depth-1, 0, -1):
            
            l_next = self.layers[depth+1]
            l:layers.TrainableLayer = self.trainable_layers[depth-1]
            l_prev = self.layers[depth-1]
            
            
            l.backpropagate(LR, momentum_coeff, num_examples_curr_batch, l_prev, l_next)
            
            # if depth > 1:
            #     dActivation = net.layers[depth-1].dActivationFunction
            #     l_min.gradE_Z = (l.weights.T @ l.gradE_Z) * dActivation(l_min.z)
                            
            
        
    def compute_loss(self, y_train_target_one_hot, Y_pred):
         
        if self.loss_type == "mse":
            assert False, "mse not implemented"
        #     # For MSE, can still vectorize
        #     Y_pred = net.layers[net.max_depth].activation
        #     Y_true = np.column_stack([ex.target_one_hot for ex in net.currBatch])
        #     loss = np.mean(0.5 * np.sum((Y_pred - Y_true)**2, axis=0))
        
        if self.loss_type == "crossentropy":
            
            # net.feedforward(batch)
            # Y_pred = net.outputLayer.activation
            Y_true = np.column_stack(y_train_target_one_hot)
            # clip to avoid log(0)
            Y_pred_clipped = np.clip(Y_pred, 1e-15, 1.0)
            # cross-entropy: sum over classes, mean over examples
            loss = -np.mean(np.sum(Y_true * np.log(Y_pred_clipped), axis=0))
                
        return loss

    def sanity_check(self, check_type="train", x_train=None, y_train_one_hot=None):
        print("\nSanity check...")
        
        n_ex = len(x_train) #type: ignore
        
        # 1. Layer structure
        assert len(self.layers) >= 2, "Network must have at least input and output layers."
        assert hasattr(self, "max_depth"), "self.max_depth not set."
        assert self.max_depth == len(self.layers) - 1, "self.max_depth mismatch."
        
        for depth, layer in enumerate(self.layers):
            if depth == 0:
              assert isinstance(layer, layers.InputLayer)
            elif depth == self.max_depth:
              assert isinstance(layer, layers.TrainableLayer)
            else:
              assert isinstance(layer, layers.TrainableLayer)
            if depth > 0:

                act = self.trainable_layers[depth-1].activation_type
                assert act in self.activation_types, f"Invalid activation type {act} for layer {depth}"

        # 2. Training examples
        if check_type == "train" and x_train is not None:
            assert n_ex > 0, "No training examples."
            X = np.column_stack(x_train)
            Y = np.column_stack(y_train_one_hot)#type: ignore
            assert X.shape[0] == self.input_layer.width, "Feature length mismatch"
            assert Y.shape[0] == self.layers[-1].width, "Target length mismatch"
            # print(X)
            assert np.all(np.isfinite(X)), "Non-numeric value in features"
            assert np.all(np.isfinite(Y)), "Non-numeric value in targets"

        # 3. Weights and biases
        for layer in self.trainable_layers:
            assert hasattr(layer, "weights") and hasattr(layer, "biases"), "Weights or biases missing"
            assert np.all(np.isfinite(layer.weights)), "Non-numeric weights"
            assert np.all(np.isfinite(layer.biases)), "Non-numeric biases"

        # 4. Forward pass for all examples
        if check_type == "train" and x_train is not None:
            self.inference_feedforward(x_train[:10000])
            output = self.output_layer.activation
            assert output.shape == (self.output_layer.width, min(10000, n_ex)), "Output shape mismatch"
            if self.output_layer.activation_type == "softmax":
                totals = np.sum(output, axis=0)
                assert np.allclose(totals, 1.0), "Softmax outputs do not sum to 1"

        print("Sanity check passed!\n")



# if __name__ == "__main__":
    
#     params = {"loss": "crossentropy", #makes output activation softmax automatically
#               "momentum": .25,
#               "layers": [2, ((2, 4), "relu"), ((4, 2), "softmax")],
#               }
    
#     net = NeuralNet(params)
    
#     training = [([0,0], [1,0]),
#                 ([0,1], [0,1]),
#                 ([1,0], [0,1]),
#                 ([1,1], [1,0]),
#                 ]

#     trainParams = {"train_data": training,
#                    "epochs": 100000,
#                    "LR": .1,
#                    "test": True,
#                    "%": True,
#                    "visual": True,
#                 #    "batch_size": 10,
#                 #    "min_accuracy": 99
#                    }
    
#     net.train(trainParams)
    
#     input("continue: ")
#     for ex in training:
#         ex = TrainingExample(*ex)
#         print(f"Features: {ex.features}")
#         print(f"Target: {ex.target_one_hot}")
#         # print(ex.target_readable)
#         out = net.feedforward([ex])
#         print(f"Output: {np.array2string(out, precision=5, separator=', ', suppress_small=True)}\n")
