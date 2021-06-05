import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tobamovirus = pd.read_csv('D:/statistics/CUHK_PPCA/data/tobamovirus.csv')
data = np.array(tobamovirus, dtype=float).T

t = data
d, N = data.shape
q = 2
K = 3

Pi = np.zeros(K)
record = np.ones(K)
Mu = np.zeros((d, K))
sigma_square = np.zeros(K)
P = np.zeros((N, K))
R = np.random.random((N, K))

sum = np.sum(R, axis = 1)
R = R / sum.reshape(N, 1)

epsilon = 0.001

while np.sqrt(np.sum(np.square(record - Pi))) >= epsilon:

    for k in range(K):
        record[k] = Pi[k]

        Pi[k] = np.sum(R[:, k]) / N
        Mu[:, k] = (np.dot(t, R[:, k]) / np.sum(R[:, k]))

        S = np.zeros((d, d))
        for n in range(N):
            S += R[n, k] * np.dot((t[:, n].reshape(d, 1) - Mu[:, k].reshape(d, 1)), (t[:, n].reshape(d, 1) - Mu[:, k].reshape(d, 1)).T)
        S /= (Pi[k] * N)

        value, U = np.linalg.eig(S)
        U = U[:, 0: q]
        sigma_square[k] = np.sum(value[q: d]).real / (d - q)
        Lambda = np.diag(value[0: q])

        names1 = 'W' + str(k)
        locals()['W' + str(k)] = np.dot(U, np.sqrt(Lambda - sigma_square[k] * np.identity(q))).real
        names2 = 'C' + str(k)
        locals()['C' + str(k)] = sigma_square[k] * np.identity(d) + np.dot(locals()['W' + str(k)], locals()['W' + str(k)].T)

        for n in range(N):
            P[n, k] = np.power(2 * np.pi, -d / 2) * (1 / np.sqrt(np.linalg.det(locals()['C' + str(k)]))) * \
                      np.exp(-np.dot((t[:, n].reshape(1, d) - Mu[:, k].reshape(1, d)),
                      np.dot(np.linalg.inv(locals()['C' + str(k)]), (t[:, n].reshape(d, 1) - Mu[:, k].reshape(d, 1)))) / 2)

    for n in range(N):
        for k in range(K):
            R[n, k] = P[n, k] * Pi[k] / (np.dot(P[n, :], Pi))

    L = 0
    for n in range(N):
        L += np.log(np.dot(P[n, :], Pi))
    print('the updated log-likelihood is:', L)

print('\n', 'Pi = ', Pi, '\n', 'R = ', '\n', R)

t0 = t[:, np.where(np.argmax(R, axis=1) == 0)].reshape(d, -1)
M0 = np.dot(W0.T, W0) + sigma_square[0] * np.identity(q)
X0 = np.dot(np.linalg.inv(M0), np.dot(W0.T, (t0 - Mu[:, 0].reshape(d, 1))))

plt.scatter(X0[0, :], X0[1, :], c = 'w')
for i in range(X0.shape[1]):
    plt.text(X0[0, i], X0[1, i], np.where(np.argmax(R, axis=1) == 0)[0][i])
plt.show()

t1 = t[:, np.where(np.argmax(R, axis=1) == 1)].reshape(d, -1)
M1 = np.dot(W1.T, W1) + sigma_square[1] * np.identity(q)
X1 = np.dot(np.linalg.inv(M1), np.dot(W1.T, (t1 - Mu[:, 1].reshape(d, 1))))

plt.scatter(X1[0, :], X1[1, :], c = 'w')
for i in range(X1.shape[1]):
    plt.text(X1[0, i], X1[1, i], np.where(np.argmax(R, axis=1) == 1)[0][i])
plt.show()

t2 = t[:, np.where(np.argmax(R, axis=1) == 2)].reshape(d, -1)
M2 = np.dot(W2.T, W2) + sigma_square[2] * np.identity(q)
X2 = np.dot(np.linalg.inv(M2), np.dot(W2.T, (t2 - Mu[:, 2].reshape(d, 1))))

plt.scatter(X2[0, :], X2[1, :], c = 'w')
for i in range(X2.shape[1]):
    plt.text(X2[0, i], X2[1, i], np.where(np.argmax(R, axis=1) == 2)[0][i])
plt.show()