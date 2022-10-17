import numpy as np 
import matplotlib.pylab as plt 
from scipy.special import softmax 
from sklearn.metrics import log_loss as cross_entropy_loss 

class MultiLayerNeuralNetwork_softmax:

    def __init__(self, in_features=100, eta=0.01, num_iter=10000):
        self.in_features = in_features
        self.eta = eta
        self.num_iter = num_iter        
        self.out_features = None
        self.num_data = None

        self.num_of_layer = 0

        self.W = {}
        self.b = {}
        self.act_fun = {}

        # Forward
        self.loss = None
        self.e = None
        self.Y = None
        self.Y_hat = None
        self.X = None
        self.a = {}
        self.z = {}
        self.input = {} # 역전파에서 이전 입력을 읽어올 때 사용합니다.
        self.output = {} # 순전파에서 이전 출력을 읽어올 때 사용합니다.
        

        # Backward
        # 역전파 및 gradient descent에 사용할 정보를 dict 형태로 저장합니다.
        self.delta = {}
        self.grad_W = {}
        self.grad_b = {}
        self.dLdW = {}
        self.dLdb = {}


        # 활성화 함수들과 그에 대한 미분을 정의하고 있습니다.
        # activation_functions
        self.activation_functions = {}
        self.activation_functions['sigmoid'] = lambda z: 1 / (1 + np.exp(-z))
        self.activation_functions['tanh'] = lambda z: np.tanh(z)
        self.activation_functions['softmax'] = lambda z : softmax(z, axis=1)

        # activation_functions_prime
        self.activation_functions_prime = {}
        self.activation_functions_prime['sigmoid'] = lambda a: a * (1 - a)
        self.activation_functions_prime['tanh'] = lambda a: (1 - np.tanh(a)) * (1 + np.tanh(a))
        self.activation_functions_prime['softmax'] = lambda y_hat, y: (y_hat - y) / y.shape[0]

    def addLayer(self, num_of_node=10, activation='sigmoid'):

        self.num_of_layer += 1

        if self.num_of_layer == 1:
            # 처음 레이어를 추가합니다.
            num_of_input = self.in_features
        else:
            # 이전 레이어의 노드 개수를 가져옵니다.
            num_of_input = self.b[self.num_of_layer - 1].shape[1]
        #############################################################################
        ##                추가된 은닉층에 대한 W, b 차원 구하기                    ##
        #############################################################################
        # 추가된 레이어가 가지는 W, b, 활성화 함수를 저장합니다.
        # None 부분을 채워주시면 됩니다.
        self.W[self.num_of_layer] = np.random.randn(num_of_input, num_of_node)
        self.b[self.num_of_layer] = np.random.randn(1, num_of_node)
        self.act_fun[self.num_of_layer] = activation
        #############################################################################
        ##                              코드 구현 끝                               ##
        #############################################################################

    def forward(self):
        # 추가된 레이어를 모두 거치며 순전파가 시작됩니다.
        for i in range(1, self.num_of_layer + 1):
            if i == 1:
                # 첫 레이어에서는 입력 데이터로 X가 사용됩니다.
                X = self.X
            else:
                # 첫 레이어가 아닌 경우, 이전 레이어의 출력이 입력데이터가 됩니다.
                X = self.output[i - 1]
            #############################################################################
            ##                               순전파 구현                               ##
            #############################################################################
            # None 부분을 채워주시면 됩니다.
            # load parameter : 현재 레이어의 W, b와 활성화함수를 읽어옵니다.
            W = self.W[i]
            b = self.b[i]
            act_fun = self.activation_functions[self.act_fun[i]]

            # Forward
            z = X.dot(W) + b
            a = act_fun(z)

            # save for backward : backpropagation에 사용될 정보를 저장합니다.
            self.z[i] = z
            self.a[i] = a
            self.output[i] = a
            self.input[i] = X
            #############################################################################
            ##                              코드 구현 끝                               ##
            #############################################################################
        # 최종 결과 Y_hat과 loss를 저장합니다.
        Y_hat = a
        loss = cross_entropy_loss(self.Y, Y_hat)
        # save for backward
        self.Y_hat = a
        self.loss = loss

    def backward(self):
        # 역전파는 출력단에서부터 계산이 시작됩니다.
        eta = self.eta
        for i in reversed(range(1, self.num_of_layer + 1)):
            # Load : 역전파에 사용할 값들을 읽어옵니다.

            W = self.W[i]
            b = self.b[i]
            a = self.a[i]
            z = self.z[i]
            act_fun = self.act_fun[i]
            N = self.num_data

            #############################################################################
            ##                               역전파 구현                               ##
            #############################################################################
            # None 부분을 채워주시면 됩니다.
            # 역전파에서는 출력층과 은닉층이 다르게 계산됩니다.
            # 출력층 
            if i == self.num_of_layer:
                dLde = self.Y - self.Y_hat
                deda = -1 * np.ones((N, 1))
                dadz = self.activation_functions_prime[act_fun](a)
                delta = dLde * deda * dadz
                delta = (-1) * delta
                dzdW = self.input[i]
                dzdb = np.ones((N, 1))
                dLdW = dzdW.T.dot(delta)
                dLdb = dzdb.T.dot(delta)

                # 은닉층
            else:
                dadz = self.activation_functions_prime[act_fun](a)
                delta = dadz * self.delta[i + 1].dot(self.W[i + 1].T)  # local gradient
                dzdW = self.input[i]
                dzdb = np.ones((N, 1))
                dLdW = dzdW.T.dot(delta)
                dLdb = dzdb.T.dot(delta)

            # Save : 다음 역전파 및 Gradient descent에 사용할 값들을 저장합니다.
            self.delta[i] = delta
            self.dLdW[i] = dLdW
            self.dLdb[i] = dLdb

        # Gradient Descent
        for i in range(1, self.num_of_layer + 1):
            # load : 편미분 값을 읽어옵니다.
            dLdW = self.dLdW[i]
            dLdb = self.dLdb[i]
            # Save : 업데이트 후 값을 저장합니다.
            self.W[i] = self.W[i] + eta * dLdW
            self.b[i] = self.b[i] + eta * dLdb
        #############################################################################
        ##                              코드 구현 끝                               ##
        #############################################################################

    def fit(self, X, Y):
        #############################################################################
        ##                                학습 구현                                ##
        #############################################################################
        # None 부분을 채워주시면 됩니다.
        # 학습 시키는 과정은 순전파와 역전파가 iterative하게 동작합니다.
        self.X = X
        self.Y = Y
        self.num_data = X.shape[0]
        for i in range(self.num_iter):
            self.forward()
            self.backward()
        #############################################################################
        ##                              코드 구현 끝                               ##
        #############################################################################

    def predict(self, x, prob=False):
        #############################################################################
        ##                     classification predict 구현                         ##
        #############################################################################
        # None 부분을 채워주시면 됩니다.
        # Predict는 forward와 다르게, 계산된 정보를 저장하지 않고, 최종 결과를 출력합니다.
        # 이 때, 결과는 예측한 class가 됩니다.

        for i in range(1, self.num_of_layer + 1):
            if i == 1:
                # 첫 레이어에서는 입력 데이터로 X가 사용됩니다.
                X = x
            else:
                # 첫 레이어가 아닌 경우, 이전 레이어의 출력이 입력데이터가 됩니다.
                X = self.output[i - 1]

            # load parameter : 현재 레이어의 W, b와 활성화함수를 읽어옵니다.
            W = self.W[i]
            b = self.b[i]
            act_fun = self.activation_functions[self.act_fun[i]]

            # forward
            z = X.dot(W) + b
            a = act_fun(z)

        #############################################################################
        ##                              코드 구현 끝                               ##
        #############################################################################
        # predict는 확률값 또는 class를 출력합니다.
        Y_hat = a
        if prob:
            print(np.round(Y_hat, 2))
            return np.round(Y_hat, 2)
        else:
            print(Y_hat.argmax(axis=1))
            return Y_hat.argmax(axis=1)


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    Y = get_one_hot(Y.reshape(-1), 2)

    MLNN = MultiLayerNeuralNetwork_softmax(2, 0.1)
    MLNN.addLayer(2)
    MLNN.addLayer(3)
    MLNN.addLayer(2)

    MLNN.fit(X, Y)
    MLNN.predict(X, prob=False)
    MLNN.predict(X, prob=True)

