#Example using general data
path = "/code_R" 
setwd(path)  
source('PPCA.R')
source('PCA.R')

a = matrix(rnorm(120), nrow = 6, ncol = 20)
b = PPCA(a, q = 3)
c = PPCA(a, q = 3, epsilon = 0.001, method = 'EM')
