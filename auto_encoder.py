import numpy                as np
import matplotlib.pyplot    as plt

class AE:

    def __init__(self, cnf, log):
        self.cnf = cnf
        self.log = log
        self.Y = np.zeros((self.cnf.N, self.cnf.out_lay))   # output


    def initialization(self):
        self.hid_weight      = self.cnf.rd.rand(self.cnf.hid_lay, self.cnf.inp_lay+1)
        self.out_weight      = self.cnf.rd.rand(self.cnf.out_lay, self.cnf.hid_lay+1)
        self.hid_momentum    = np.zeros((self.cnf.hid_lay, self.cnf.inp_lay+1))
        self.out_momentum    = np.zeros((self.cnf.out_lay, self.cnf.hid_lay+1))

    def training(self):
        self.error = np.zeros(self.cnf.max_epoch)
        for epo in range(self.cnf.max_epoch):
            for i in range(self.cnf.N):
                x = self.cnf.X[i, :]
                self.__update_weight(x)
            self.error[epo] = self.__calc_error(self.cnf.X)
            self.log.logging(epo+1, self.error[epo], self.hid_weight, self.out_weight)
            if epo % 100 == 99:
                self.log.logging_detail(epo+1, self.Y, self.error[epo])
        self.log.outLog()

    def predict(self):
        err =  self.__calc_error(self.cnf.test_data)
        self.log.logging_predict(self.Y, err)

    def out_errorgraph(self):
        plt.xlim({0, self.cnf.max_epoch})
        plt.plot(np.arange(0, self.error.shape[0]), self.error)
        plt.show()



    def __sigmoid(self, arr):
        return np.vectorize(lambda x: 1.0 / (1.0 + np.exp(-x)))(arr)

    def __forward(self, x):
        # output in hidden layer
        z = self.__sigmoid(self.hid_weight.dot(np.r_[np.array([1]), x]))
        # output in output layer
        y = self.__sigmoid(self.out_weight.dot(np.r_[np.array([1]), z]))
        return (z, y)

    def __update_weight(self, x):
        z, y = self.__forward(x)

        # update output_weight
        out_delta = (y - x) * y * (1.0 - y)
        _out_weight = self.out_weight
        self.out_weight -= self.cnf.epsilon * out_delta.reshape((-1, 1)) * np.r_[np.array([1]), z] - self.cnf.mu * self.out_momentum
        self.out_momentum = self.out_weight - _out_weight

        # update hidden_weight
        hid_delta = (self.out_weight[:, 1:].T.dot(out_delta)) * z * (1.0 - z)
        _hid_weight = self.hid_weight
        self.hid_weight -= self.cnf.epsilon * hid_delta.reshape((-1, 1)) * np.r_[np.array([1]), x]
        self.hid_momentum = self.hid_weight - _hid_weight

    def __calc_error(self, X):
        for i in range(self.cnf.N):
            x = X[i, :]
            z, y = self.__forward(x)
            self.Y[i] = y
        return np.sum((self.Y - X) ** 2) / 2.