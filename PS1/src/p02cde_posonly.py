import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path,label_col='t', add_intercept=True)
    x_test,t_test = util.load_dataset(test_path,label_col='t', add_intercept=True)
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    
    model1 = LogisticRegression()
    model1.fit(x_train,t_train)
    pred1 = model1.predict(x_test)

    accuracy = np.mean(pred1 == t_test)
    print(f"Accuracy {accuracy:.4f}")
    np.savetxt(pred_path_c, pred1)
    
    # Part (d): Train on y-labels and test on true labels

    x_train2,y_train2 =util.load_dataset(train_path,label_col='y',add_intercept=True)
    x_test2,y_test2 = util.load_dataset(test_path,label_col='y',add_intercept=True)

    model2 = LogisticRegression()
    model2.fit(x_train2,y_train2)
    pred2 = model2.predict(x_test2)
    accuracy = np.mean(pred2==y_test2)

    print(f"Accuracy: {accuracy:.4f}")
    np.savetxt(pred_path_d, pred2)

    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    x_val,t_val = util.load_dataset(valid_path,label_col='t',add_intercept=True)
    scores_val = (1/(1+np.exp(-(x_val@model2.theta))))
    alpha = np.mean(scores_val[t_val==1])
    print(f"Alpha: {alpha:.4f}")

    scores_test = 1/(1+np.exp(-(x_test@model2.theta)))
    corrected = scores_test/alpha
    pred_e =(corrected>=0.5).astype(int)

    accuracy = np.mean(pred_e==t_test)
    print(f"Accuract after correction: {accuracy:.4f}")
    np.savetxt(pred_path_e,pred_e)
    
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])