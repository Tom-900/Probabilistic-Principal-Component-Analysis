import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PCA import PCA
from PPCA import PPCA


tobamovirus = pd.read_csv('D:/statistics/CUHK_PPCA/data/tobamovirus.csv')
data = np.array(tobamovirus, dtype=float).T

pca_data = PCA(data, 2)
plt.scatter(pca_data[0, :], pca_data[1, :], c = 'w', )
plt.title('PCA', fontsize='large', fontweight='bold')
for i in range(38):
    plt.text(pca_data[0, i], pca_data[1, i], i+1)
plt.show()

ppca_data = PPCA(data, 2, 0.001, method='EM')
plt.scatter(ppca_data[0, :], ppca_data[1, :], c = 'w')
plt.title('PPCA', fontsize='large', fontweight='bold')
for i in range(38):
    plt.text(ppca_data[0, i], ppca_data[1, i], i+1)
plt.show()



