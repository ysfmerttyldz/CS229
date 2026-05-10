import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    model = PoissonRegression(step_size=float(lr),eps=1e-5,max_iter=10000)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    np.savetxt(pred_path,y_pred)
    plt.figure()
    plt.plot(y_eval,y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')


class PoissonRegression(LinearModel):

    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.zeros(n)
        for i in range(self.max_iter):
            theta = np.copy(self.theta)
            self.theta += self.step_size * x.T.dot(y - np.exp(x.dot(self.theta))) / m
            if np.linalg.norm(self.theta-theta) < self.eps:
                break

    def predict(self, x):
        return np.exp(x.dot(self.theta))


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])