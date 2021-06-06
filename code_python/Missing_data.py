import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This programme is not debugged

def missing_PPCA(missing_data, q):
    # missing data is the array of numpy, with a shape of d * N
    d = missing_data.shape[0]
    N = missing_data.shape[1]
    O = np.ones((d, N))
    Y = missing_data
    data_mean = missing_data
    for i in range(d):
        for j in range(N):
            if np.isnan(missing_data[i, j]):
                O[i, j] = 0
                Y[i, j] = 0
                data_mean[i, j] = np.nanmean(missing_data, axis=1)[i]

    m = np.mean(data_mean, axis=1).reshape(d, 1)
    S = np.zeros((d, d))
    for i in range(N):
        S += np.dot((data_mean[:, i].reshape(d, 1) - m), (data_mean[:, i].reshape(d, 1) - m).T)
    S /= N

    value, U = np.linalg.eig(S)
    U = U[:, 0: q]
    v = np.sum(value[q: d]) / (d - q)
    Lambda = np.diag(value[0: q])
    W = np.dot(U, np.sqrt(Lambda - v * np.identity(q)))
    W = W.real

    M = np.dot(W.T, W) + v * np.identity(q)
    A = np.dot(np.dot(np.linalg.inv(M), W.T), (data_mean - m))


    X = np.zeros((q, N))
    record = 1
    epsilon = 0.001

    while np.abs(record - v) >= epsilon:
        record = v
        # E step
        for j in range(N):
            T = O[:, j].reshape(d, 1)
            for k in range(q - 1):
                T = np.hstack((T, O[:, j].reshape(d, 1)))
            W_j = W * T

            M_j = m * O[:, j].reshape(d, 1)

            Psi = np.dot(W_j.T, W_j) + v * np.identity(q)
            X[:, j] = np.dot(np.linalg.inv(Psi), np.dot(W_j.T, (Y[:, j].reshape(d, 1) - M_j))).reshape(q)
            name = 'Sigma' + str(j)
            locals()['Sigma' + str(j)] = v * np.linalg.inv(Psi)


        # M step
        for i in range(d):
            X_i = X
            m[i] = 0
            W[i, :] = 0
            C = np.zeros((q, q))
            v = 0
            for j in range(N):
                T = O[i, :].reshape(1, N)
                for k in range(q - 1):
                    T = np.vstack((T, O[i, :].reshape(1, N)))
                    X_i = X * T

                if O[i, j] == 1:
                    m[i] = m[i] + Y[i, j] - np.dot(W[i, :].reshape(1, q), X[:, j].reshape(q, 1))
                    C += locals()['Sigma' + str(j)]
                    v += (np.square(Y[i, j] - np.dot(W[i, :].reshape(1, q), X[:, j].reshape(q, 1)) - m[i])
                          + np.dot(W[i, :].reshape(1, q), np.dot(locals()['Sigma' + str(j)], W[i, :].reshape(q, 1)))) / N
            W[i, :] = np.dot((Y[i, :].reshape(N, 1) - m[i]).T, np.dot(X_i.T, np.linalg.inv(np.dot(X_i, X_i.T) + C)))
            m[i] /= np.sum(O[i, :])
        v = float(v)

    X = A
    return X


random_matrix = np.random.random((18, 38))
for i in range(18):
    for j in range(38):
        if random_matrix[i, j] <= 0.2:
            random_matrix[i, j] = np.nan
        else:
            random_matrix[i, j] = 1

tobamovirus = pd.read_csv('/data/tobamovirus.csv')
data = np.array(tobamovirus, dtype=float).T

missing_data = data * random_matrix

data_imputed = missing_PPCA(missing_data, 2)
print(data_imputed)
plt.scatter(data_imputed[0, :], data_imputed[1, :], c='w')
for i in range(38):
    plt.text(data_imputed[0, i], data_imputed[1, i], i+1)
plt.show()










