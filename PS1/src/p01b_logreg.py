import numpy as np
import util

from linear_model import LinearModel




def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train,y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    pred = model.predict(x_eval)
    np.savetxt(pred_path,pred,fmt = "%d")
    
    accuracy = np.mean(pred == y_eval)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Theta: {model.theta}")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def h(self,x):
        z = np.dot(x,self.theta)
        return 1/(1+np.exp(-z))

    def log_likelihood(self,x,y):
        h = self.h(x)
        return np.sum(y*np.log(h)+(1-y)*np.log(1-h))
    
    def gradient(self,x,y):
        h = self.h(x)
        return np.dot(x.T,(y-h))

    def hessian(self,x,y):
        m = len(y)
        h = self.h(x)
        s = (h*(1-h))
        w = np.diag(s)
        return (1/m)*(np.dot(np.dot(x.T,w),x))      

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        if(self.theta is None):
            self.theta = np.zeros(n)
        
        while True:
            theta_prev = np.copy(self.theta)
            h = self.h(x)
            gradient = self.gradient(x,y)
            hessian = self.hessian(x,y)
            gradient_j = -(1/m)*gradient
            self.theta -= np.dot(np.linalg.inv(hessian),gradient_j)
            if(np.sum(np.abs(self.theta-theta_prev))<self.eps):
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        h = self.h(x)
        return (h>=0.5).astype(int)
        # *** END CODE HERE ***


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])