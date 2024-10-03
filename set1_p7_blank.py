import numpy as np
import matplotlib.pyplot as plt
# from sympy import diff
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
    #w = np.transpose(W)
    prediction = np.dot(X, W)
    loss = np.sum(np.power(prediction - Y, 2)) / (2 * N)
    loss = np.log(loss)
    # -------------------------------------------------
    return loss

def gradL(X, Y, W):
    """
    Computes the gradient of L(W)
    """
    # ---------- make your implementation here -------------
    step = 0.0001
    grad = np.zeros_like(W)
    for i in range(len(W)):
        W_eps = np.copy(W)
        W_eps[i] += step
        grad[i] = (Loss(W_eps, X, Y) - Loss(W, X, Y)) / step
    return grad
    # loss = Loss(W, X, Y)
    # print (loss)
    # lossH = Loss(W + step, X, Y) # 0.0001 is the step size
    # print (lossH)
    # grad = (lossH - loss) / step
    # return grad
    # -------------------------------------------------

def hessian(X, N, W):
    """
    Computes the Hessian matrix
    """
    h_w = []
    # ---------- make your implementation here -------------
    # h_w = np.zeros((3,3))
    # for i in range(N):
    #     elem = sig(X[i]) * (1 - sig(X[i])) * np.outer(X[i], X[i])
    #     h_w += elem
    # # print(h_w)
    # return h_w
    Z = np.dot(X, W)
    predictions = sig(Z)
    D = np.diag(predictions * (1 - predictions)) 
    return np.dot(X.T, np.dot(D, X))
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
        N = Ytrain.shape[0]
        # for i in range(N):
        H = hessian(Xtrain, Ytrain.shape[0], W)
        G = gradL(Xtrain, Ytrain, W)
        print(H)
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
plt.plot(range(79), train_loss, '-r^', range(79), test_loss, '-bo')
plt.legend(['train loss', 'test loss'])
plt.show()