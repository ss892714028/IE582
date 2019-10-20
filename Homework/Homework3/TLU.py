import numpy as np


class TLU:
    def __init__(self, x, y, alpha, epoch, theta=0.5):
        self.x = x
        self.y = y
        self.feature_dim = self.x.shape[1]
        self.theta = theta
        self.alpha = alpha
        self.epoch = epoch

    @staticmethod
    def add_feature(d):
        data = []
        for index, value in enumerate(d):
            data.append(np.append(d[index], -1))
        return np.array(data)

    def fit(self):
        self.x = self.add_feature(self.x)
        x = self.x
        y = self.y
        w = np.zeros(self.x.shape[1])
        w[-1] = self.theta
        for epochs in range(self.epoch):
            for i in range(x.shape[0]):
                sample = x[i]
                y_hat = self.predict(sample, w)
                w += self.alpha * (y[i] - y_hat) * sample
                print('step: {} | weights: {} | theta: {}'.format(i, w[:-1], round(w[-1], 3)))
        return w

    @staticmethod
    def predict(x, w):
        dot = np.dot(w, x)
        if dot >= 0:
            y_hat = 1
        else:
            y_hat = 0
        return y_hat

    def test(self, w):
        x = self.x
        y = self.y
        error = 0
        for i in range(self.x.shape[0]):
            y_hat = self.predict(x[i], w)
            if y_hat != y[i]:
                error += 1
        accuracy = 1-error/len(y)
        return accuracy
