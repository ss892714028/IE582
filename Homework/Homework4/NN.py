import numpy as np
import pandas as pd


class NN:
    def __init__(self, x, y, units=3, lr=0.1):
        self.x = x
        self.y = y
        self.wh = np.random.rand(self.x.shape[1], units)
        self.w0 = np.random.rand(units, 1)
        self.b0 = np.random.rand(1)
        self.bh = np.random.rand(units)
        self.lr = lr

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_dev(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        wh, w0, b0, bh = self.wh, self.w0, self.b0, self.bh

        zh = np.dot(x, wh)
        ah = self.sigmoid(zh) + bh

        z0 = np.dot(ah, w0) + b0
        a0 = self.sigmoid(z0)
        return zh, ah, z0, a0

    def backward(self, x, y):

        zh, ah, z0, a0 = self.forward(x)
        ah = ah.reshape(-1,1)
        y_hat = a0[0]
        loss = - y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

        self.w0 -= self.lr * np.multiply((y_hat - y), ah).reshape(-1,1)
        self.b0 -= self.lr * (y_hat - y)
        self.bh -= np.array(self.lr * (y_hat - y) * self.w0 * ah * (1 - ah)).reshape(-1)
        for i in range(self.wh.shape[0]):
            self.wh[i] -= np.array(self.lr * (y_hat - y) * self.w0 * ah * (1 - ah) * x[i]).reshape(-1)
        print(loss)


if __name__ == '__main__':
    data = pd.read_csv('or.csv', header=None).to_numpy()
    x = data[:, :2]
    y = data[:, 2]
    nn = NN(x, y)
    epochs = 50
    for epoch in range(epochs):
        for i in range(x.shape[0]):
            nn.backward(x[i], y[i])