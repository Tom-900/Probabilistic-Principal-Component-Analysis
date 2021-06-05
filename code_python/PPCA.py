import numpy as np

def PPCA(t, q, epsilon = 0.001, method = 'ML'):
    d, N = t.shape
    Mu = np.mean(t, axis = 1).reshape(d, 1)
    S = np.zeros((d, d))

    for i in range(N):
        S += np.dot((t[:, i].reshape(d, 1) - Mu), (t[:, i].reshape(d, 1) - Mu).T)
    S /= N

    if method == 'ML':
        value, U = np.linalg.eig(S)
        U = U[:, 0: q]
        sigma_square = np.sum(value[q: d]) / (d - q)
        Lambda = np.diag(value[0: q])
        W = np.dot(U, np.sqrt(Lambda - sigma_square * np.identity(q)))
        W = W.real

        M = np.dot(W.T, W) + sigma_square * np.identity(q)
        X = np.dot(np.dot(np.linalg.inv(M), W.T), (t - Mu))

    if method == 'EM':
        value, U = np.linalg.eig(S)
        U = U[:, 0: q]
        U = U.real
        Lambda = np.diag(value[0: q])

        W_new = np.dot(U, np.sqrt(Lambda))
        W_old= np.dot(U, np.sqrt(Lambda))
        W_new = W_new.real
        W_old = W_old.real
        sigma_square_new = 1
        sigma_square_old = 5

        while (np.sqrt(np.sum(np.square(W_new - W_old))) > epsilon) | (abs(sigma_square_new-sigma_square_old) > epsilon):
            M = np.dot(W_new.T, W_new) + sigma_square_new * np.identity(q)
            W_old = W_new
            sigma_square_old = sigma_square_new
            W_new = np.dot(np.dot(S, W_old), np.linalg.inv(sigma_square_old * np.identity(q) + np.dot(np.linalg.inv(M), np.dot(W_old.T, np.dot(S, W_old)))))
            sigma_square_new = np.trace(S - np.dot(S, np.dot(W_old, np.dot(np.linalg.inv(M), W_new.T))))/d

            C = np.dot(W_new, W_new.T) + sigma_square_new * np.identity(d)
            L = -N * (d * np.log(np.pi) + np.log(np.linalg.det(C)) + np.trace(np.dot(np.linalg.inv(C), S))) / 2
            print('updated log-likelihood: ', L)

        M = np.dot(W_new.T, W_new) + sigma_square_new * np.identity(q)
        X = np.dot(np.linalg.inv(M), np.dot(W_new.T, (t - Mu)))

    return X








