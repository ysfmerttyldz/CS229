import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    
    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train,y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    pred = model.predict(x_eval)
    accuracy = np.mean(pred == y_eval)
    print(f"Accuract: {accuracy:.4f}")
    print(f"Theta: {model.theta}")
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        phi = np.sum(y)/m
        mu0 = np.mean(x[y==0],axis=0)
        mu1 = np.mean(x[y==1],axis=0)

        sigma = np.zeros((n,n))
        for i in range(m):
            diff = 0
            if(y[i]==0):
                diff = x[i]-mu0
            else:
                diff = x[i]-mu1
            diff = diff.reshape(-1,1)
            sigma+= np.dot(diff,diff.T)
        sigma /= m
        sigma_inv = np.linalg.inv(sigma)
        theta_vec = np.dot(sigma_inv,(mu1-mu0))
        theta_0 = 0.5 * (mu0 @ sigma_inv @ mu0 - mu1 @ sigma_inv @ mu1)
        theta_0 += np.log(phi/(1-phi))

        self.theta = np.zeros(n+1)
        self.theta[0] = theta_0
        self.theta[1:] = theta_vec

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        newx = np.c_[np.ones(x.shape[0]),x]
        return (1/(1+np.exp(-(newx @ self.theta)))> 0.5).astype(int)
        # *** END CODE HERE

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])