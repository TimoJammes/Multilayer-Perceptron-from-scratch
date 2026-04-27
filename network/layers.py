import numpy as np
from scipy.signal import convolve2d

LEAKY_ALPHA = 0.01
INIT_BIAS = 0.01
EPS = 1e-8


def sigmoid(x):
    
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    
    s = sigmoid(x)
    return s*(1-s)

def ReLU(x):
    
    return np.maximum(0, x)

def d_ReLU(X):
    return np.where(X <= 0, 0, 1)

def leaky_ReLU(x):
    return np.where(x >= 0, x, LEAKY_ALPHA * x)

def d_leaky_ReLU(x):
    return np.where(x >= 0, 1.0, LEAKY_ALPHA)

def softmax(x):
    # Z = self.sum   
    Z = x               
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)  
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


class Layer:
    def __init__(self, depth, width):
        self.depth = depth
        self.width = width
        self.output = None

    def feed(self, input):
        raise NotImplementedError
    
class InputLayer(Layer):
    
    def __init__(self, width):
        super().__init__(depth=0, width=width)
        # self.width = width
        
        # self.depth = 0
        
        # self.sum = np.zeros((1, width))
        # self.activation = self.sum
        self.output = np.zeros((1, width))

    def feed(self, input):
        
        # self.inputLayer.sum = input.T
        self.output = input

        # self.activation = self.output
        # self.inputLayer.activation = self.inputLayer.sum
        
        
        
        # return self.output
    
class TrainableLayer(Layer):
    
    activation_to_func = {"relu": ReLU,
                            "leakyrelu": leaky_ReLU,
                            "sigmoid": sigmoid,
                            "softmax": softmax
                            }
    activation_to_d_func = {"relu": d_ReLU,
                            "leakyrelu": d_leaky_ReLU,
                            "sigmoid": d_sigmoid,
                            "softmax": None
                            }

    BN_alpha = 0.9
    
    def __init__(self, depth, in_width, width, activation, do_batch_norm, is_output=False):
        super().__init__(depth=depth, width=width)
        # self.depth = depth
        # self.width = width
        # self.neurons = []
        
        limit = np.sqrt(2 / in_width) #HE INITIALIZATION
        
        # self.weights = np.array([[dist(-limit, limit) 
        #                                 for i in range(inWidth)] 
        #                                for i in range(width)])
        
        self.weights = np.random.uniform(-limit, limit, (width, in_width))
        # self.biases = np.reshape(np.array([INIT_BIAS for i in range(width)]), (-1, 1))
        self.biases = np.full((width, 1), INIT_BIAS)

        
        self.weights_velocity = np.zeros_like(self.weights)
        self.biases_velocity = np.zeros_like(self.biases)
        
        assert activation!="softmax" or is_output, "softmax activation only valid for output layer!"
        
        self.activation_type = activation
        self.activation_function = self.activation_to_func[activation]
        self.d_activation_function = self.activation_to_d_func[activation]
        
        self.do_batch_norm = do_batch_norm
        
        if self.do_batch_norm:
            self.BN_gamma_velocity = np.zeros_like(self.biases)
            self.BN_beta_velocity = np.zeros_like(self.biases)
            
            self.BN_moving_avg_std = np.ones_like(self.biases)
            self.BN_moving_avg_mean = np.zeros_like(self.biases)
            
            self.BN_gamma = np.ones_like(self.biases)
            self.BN_beta = np.zeros_like(self.biases)
    
    def inference_feed(self, input):
        
        self.calc_z(input)
        
        activation_input = self.z
        if self.do_batch_norm:
            
            activation_input = (activation_input - self.BN_moving_avg_mean) / (self.BN_moving_avg_std+EPS)
            activation_input = self.BN_gamma * activation_input + self.BN_beta
                        
        self.calc_activation(activation_input)
        
        self.output = self.activation
        
    def training_feed(self, input):
        
        self.calc_z(input)
        
        activation_input = self.z
        if self.do_batch_norm:
            
            BN_input = self.z
            self.batch_normalize(BN_input)
            
            activation_input = self.BN_output
            
        self.calc_activation(activation_input)
        
        self.output = self.activation
        
        # return self.output
    
    def calc_z(self, in_activation):
        
        self.z = self.weights @ in_activation
        self.z += self.biases
    
    def batch_normalize(self, input):
        
        mean = np.mean(input, axis=1, keepdims=True)
        self.BN_std = np.std(input, axis=1, keepdims=True)
        
        self.BN_moving_avg_mean = self.BN_alpha * self.BN_moving_avg_mean +  (1-self.BN_alpha)*mean
        self.BN_moving_avg_std = self.BN_alpha * self.BN_moving_avg_std +  (1-self.BN_alpha)*self.BN_std
        
        self.BN_normalized_input = (input - mean) / (self.BN_std+EPS)
    
        self.BN_output = self.BN_gamma * self.BN_normalized_input + self.BN_beta
        
    def calc_activation(self, input):
        self.activation = self.activation_function(input)
        
    def backpropagate(l, learning_rate, momentum_coeff, num_examples, prev_layer, next_layer):
        
        
        l.grad_E_wrt_A = next_layer.weights.T @ next_layer.grad_E_wrt_Z
        
        if l.do_batch_norm:
            
            l.grad_E_wrt_BN_output = l.grad_E_wrt_A * l.d_activation_function(l.BN_output)
            
            l.grad_E_wrt_BN_gamma = np.sum(l.grad_E_wrt_BN_output * l.BN_normalized_input, axis=1, keepdims=True)
            l.grad_E_wrt_BN_beta = np.sum(l.grad_E_wrt_BN_output, axis=1, keepdims=True)
            # l.gradE_wrt_BN_normalized_input = l.BN_gamma * l.gradE_wrt_BN_output
            
            l.grad_E_wrt_BN_input = 1/num_examples * l.BN_gamma * (l.BN_std**2 + EPS)**(-1/2) * (
                -l.grad_E_wrt_BN_gamma*l.BN_normalized_input 
                + num_examples*l.grad_E_wrt_BN_output
                - l.grad_E_wrt_BN_beta)
            
            l.grad_E_wrt_Z = l.grad_E_wrt_BN_input
        else:
            l.grad_E_wrt_Z = l.grad_E_wrt_A * l.d_activation_function(l.z)
        
        l.grad_E_wrt_weights = l.grad_E_wrt_Z @ (prev_layer.output).T 
        l.grad_E_wrt_biases = np.sum(l.grad_E_wrt_Z, axis=1, keepdims=True)
        
        l.weights_velocity = momentum_coeff * l.weights_velocity + l.grad_E_wrt_weights
        l.biases_velocity = momentum_coeff * l.biases_velocity + l.grad_E_wrt_biases
        
        l.weights = l.weights - learning_rate * l.weights_velocity
        l.biases = l.biases - learning_rate * l.biases_velocity
        
        if l.do_batch_norm:
            l.BN_gamma_velocity = momentum_coeff * l.BN_gamma_velocity + l.grad_E_wrt_BN_gamma
            l.BN_beta_velocity = momentum_coeff * l.BN_beta_velocity + l.grad_E_wrt_BN_beta
            
            l.BN_gamma = l.BN_gamma - learning_rate * l.BN_gamma_velocity
            l.BN_beta = l.BN_beta - learning_rate * l.BN_beta_velocity
        

# class ConvolutionLayer:
    
#     def __init__(self, depth: int, in_size: int, out_size: int, num_filters: int, num_in_channels: int, stride: int, padding:int):
        
#         self.depth = depth
#         self.in_size = in_size
#         self.out_size = out_size
#         self.num_filters = num_filters
#         self.num_in_channels = num_in_channels
#         self.stride = stride
#         self.padding = padding
        
    
#     def training_feed(self, input):
        
#         #input: IxIxC
#         I1, I2, C = input
#         assert I1 == I2
#         I = I1
#         assert I == self.in_size
#         assert C == self.num_in_channels
        
#         output = np.empty_like()
#         for slice in range(C):
#             output[:, :, slice] = convolve2d()
            
        
        
        
