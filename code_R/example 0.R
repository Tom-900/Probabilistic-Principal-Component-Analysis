#Example using general data

a = matrix(rnorm(120), nrow = 6, ncol = 20)
b = PPCA(a, q = 3)
c = PPCA(a, q = 3, epsilon = 0.001, method = 'EM')
