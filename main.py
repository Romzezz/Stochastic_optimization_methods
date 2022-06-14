#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


# In[2]:


def getBatch(X, Y, batch_size):
    batch_indxs = np.random.choice(len(X), size=batch_size, replace=False)
    X_batch = X[batch_indxs]
    Y_batch = Y[batch_indxs]
    return X_batch, Y_batch


# In[3]:


def predict(x, H_params):
    W, b = H_params
    return (np.dot(W.transpose(), x) + b)


# In[5]:


def plotHyperPlane(ax, H_params, t=None, lr_t=None):
    # plot the line, the points, and the nearest vectors to the plane
    x1min, x2min = np.min(X, axis=0)
    x1max, x2max = np.max(X, axis=0)
    xx = np.linspace(x1min, x1max, 10)
    yy = np.linspace(x2min, x2max, 10)

    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = predict([[x1], [x2]], H_params)
        Z[i, j] = p[0]
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors = 'k'
    ax.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
               edgecolor='black', s=20)


# In[81]:


def SVM_SGD(X, Y, X_new, C=0.1, plot=False):
    W = np.random.rand(2,1)
    b = np.random.rand()
    H_params = W,b
    lr=[1e-1,1e-3]
    n_iter = 200
    lr_diff = lr[1]-lr[0]
    loss="hinge"
    batch_size=50

    for t in tqdm(range(n_iter), leave=False):
        lr_t = (lr_diff/(n_iter-1))*t + lr[0]   # linear
        X_batch, Y_batch = getBatch(X,Y,batch_size)
        delI_by_delW = np.zeros(W.shape)
        delI_by_delb = 0
        # sum the hinge loss gradients by looping over each sample
        for x,y in zip(X_batch,Y_batch):
            output = y*(np.dot(W.transpose(), x) + b)
            if loss in ["hinge", "sq_hinge"]:
                if output < 1:
                    if loss == "hinge":
                        delI_by_delW += -y*np.transpose([x])
                        delI_by_delb += -y
                elif loss == "sq_hinge":  # use smaller value of C for squared hinge loss
                    delI_by_delW += 2*(1-output)*(-y*np.transpose([x]))
                    delI_by_delb += 2*(1-output)*(-y)
            elif loss == "logistic":
                delI_by_delW += (-np.exp(output)/(1+np.exp(output)))*(-y*np.transpose([x]))
                delI_by_delb += (-np.exp(output)/(1+np.exp(output)))*(-y)

        delI_by_delW = delI_by_delW*len(X)/batch_size
        delI_by_delb = delI_by_delb*len(X)/batch_size

        delf_by_delW = W + C*delI_by_delW
        delf_by_delb = C*delI_by_delb

        W = W - lr_t*delf_by_delW
        b = b - lr_t*delf_by_delb
    
    Y_new = [-1 if pred <= 0 else 1 for pred in predict(X_new.transpose(), (W, b))[0].tolist()]

    if plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("x1");  ax.set_ylabel("x2")
        plotHyperPlane(ax, (W,b), t=t,lr_t=lr_t)

    return {'W': W, 'b': b, 'Y_new': Y_new}

