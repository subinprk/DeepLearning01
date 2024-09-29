import numpy as np
import matplotlib.pyplot as plt
# Read Data
Xtest = np.loadtxt('test_features.dat')
Ytest = np.loadtxt('test_labels.dat')
Ntest = Ytest.shape[0]
Xtest = np.concatenate((np.ones((Ntest,1)), Xtest), axis=1)

Xtrain = np.loadtxt('train_features.dat')
Ytrain = np.loadtxt('train_labels.dat')
Ntrain = Ytrain.shape[0]
Xtrain = np.concatenate((np.ones((Ntrain,1)), Xtrain), axis=1)


def sig(x):
    """
    Returns the sigmoid/logistic function of input
    """
    return 1 / (1 + np.exp(-x))


def Loss(W, X, Y):
    """
    Computes the negative log likelihood function for the parameter W
    Input: Weight matrix W
    Output: loss = -log(l(W))
    """
    loss = 0
    N = Y.shape[0]
    # ---------- make your implementation here -------------

    # -------------------------------------------------
    return loss


def gradL(X, Y, W):
    """
    Computes the gradient of L(W)
    """
    # ---------- make your implementation here -------------

    # -------------------------------------------------

def hessian(X, N, W):
    """
    Computes the Hessian matrix
    """
    h_w = []
    # ---------- make your implementation here -------------

    # -------------------------------------------------

def check_accuracy(w, X, Y):
    """
    Return the accuracy
    """
    pred = 1.0 / (1 + np.exp(-np.dot(X, w.T)))

    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    accuracy = 1 - np.count_nonzero(pred - Y) / np.size(Y)

    return accuracy


def Newton(Xtrain, Ytrain, Xtest, Ytest, tol):
    """
    Performs Newton's method to update the weight matrix
    Input: X,Y, tol(tolerance)
    Output: Weight Matrix W, error
    """
    W = np.array([0.0, 0.0, 0.0])
    error = 10
    steps = 0
    train_loss = []
    test_loss = []
    while (error > tol):
        l1_train = Loss(W, Xtrain, Ytrain)
        l1_test = Loss(W, Xtest, Ytest)
        steps += 1
        # for i in range(N):
        H = hessian(Xtrain, Ytrain.shape[0], W)
        G = gradL(Xtrain, Ytrain, W)
        W = W - np.dot(np.linalg.pinv(H), G)

        l2_train = Loss(W, Xtrain, Ytrain)
        l2_test = Loss(W, Xtest, Ytest)

        error = np.abs(l2_train - l1_train)

        train_loss.append(l1_train)
        test_loss.append(l1_test)

        print('Train error: ', error)
        # print check_accuracy(W)
    return W, error, steps, train_loss, test_loss

W, error,steps, train_loss, test_loss = Newton(Xtrain, Ytrain, Xtest, Ytest, 10**(-8))
print("\nNewton's\n", W, "\n Final Loss: ", error, " Steps: ", steps)
plt.plot(range(9), train_loss, '-r^', range(9), test_loss, '-bo')
plt.legend(['train loss', 'test loss'])
plt.show()