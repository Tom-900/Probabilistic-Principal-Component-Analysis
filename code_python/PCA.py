import numpy as np

def PCA(t, q):
    d, N = t.shape

    Mu = np.mean(t, axis=1).reshape(d, 1)
    S = np.zeros((d, d))

    for i in range(N):
        S = S + np.dot((t[:, i].reshape(d, 1) - Mu), (t[:, i].reshape(d, 1) - Mu).T)
    S = S / N

    _, W = np.linalg.eig(S)
    W = W[:, 0: q].real
    X = np.dot(W.T, (t - Mu))

    return X




