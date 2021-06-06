#Example about controling the degrees of freedom

library('limSolve')
path = "D:/statistics/CUHK_PPCA/code" 
setwd(path)  
source('PPCA.R')
source('PCA.R')

tobamovirus = read.csv('D:/statistics/CUHK_PPCA/data/tobamovirus.csv')
data = t(as.matrix(tobamovirus))

#1. Compute the number of parameters
num_para = function(data, q){
  t = data
  d = dim(t)[1]
  num = d*q + 1 - q*(q-1)/2
  
  return(num)
}


#2. PPCA_2
PPCA_2 = function(data, q){
  
  t = as.matrix(data)
  d = dim(t)[1]
  N = dim(t)[2]
  Mu = rowMeans(t)

  S = matrix(data = 0, nrow = d, ncol = d)
  for (i in 1 : N){S = S + (t[, i] - Mu) %*% t(t[, i] - Mu)}
  S = S / N

  ev = eigen(S)
  
  if (q + 1 == d){sigma = sum(ev$values[d]) / (d - q)}
  if (q + 1 < d){sigma = sum(ev$values[(q + 1) : d]) / (d - q)}

  U = ev$vectors[, 1 : q]
  if (q == 1){lambda = ev$values[1]}
  if (q > 1){lambda = diag(ev$values[1 : q])}
  W = U %*% sqrt(lambda - sigma * diag(q))
  
  C = W %*% t(W) + sigma*diag(d)
  return(C)
}  


#3. Isotropic model
Isotropic = function(data, q){
  t = as.matrix(data)
  d = dim(t)[1]
  N = dim(t)[2]
  Mu = rowMeans(t)
  
  S = matrix(data = 0, nrow = d, ncol = d)
  for (i in 1 : N){S = S + (t[, i] - Mu) %*% t(t[, i] - Mu)}
  S = S / N
  sigma = sum(diag(S)) / d
  C = sigma * diag(d)
  return(C)
}


#4. Diagonal model
Diagonal = function(data, q){
  t = as.matrix(data)
  d = dim(t)[1]
  N = dim(t)[2]
  Mu = rowMeans(t)
  
  S = matrix(data = 0, nrow = d, ncol = d)
  for (i in 1 : N){S = S + (t[, i] - Mu) %*% t(t[, i] - Mu)}
  S = S / N
  
  C = diag(diag(S))
  return(C)
}


#5. compute the estimated prediction error
prediction_error = function(data, q, num_iter, f){
#Input: 
#data: d*N data matrix
#q: the dimension of the latent space
#num_iter: number of the iteration
#f: a function:Isotropic or Diagonal or myppca
#Output:
#error: the negative log likelihood
  
error_array = array(0, num_iter)
  for(i in 1 : num_iter){
    d = dim(data)[1]
    N = dim(data)[2]
    train_index = unique(sample(c(1:N), N, replace = TRUE))
    test_index = c(1:N)[-unique(sample(c(1:N), N, replace = TRUE))]
    train_data = data[, train_index]
    test_data = data[, test_index]
    
    C = f(train_data, q)
    n = dim(test_data)[2]
    train_Mu = rowMeans(train_data)
    S_test = matrix(0, d, d)
    for(j in 1 : n){S_test = S_test + (test_data[, n]-train_Mu) %*% t((test_data[, n]-train_Mu))}
    S_test = S_test / n
    L_test = -n*(log(det(C))+sum(diag(Solve(C) %*% S_test)))/2
    error_array[i] = -L_test / n
  }
  error = mean(error_array)
  
  return(error)
}


Isotropic_error=prediction_error(data, 0, 500, Isotropic)
Diagonal_error=prediction_error(data, 0, 500, Diagonal)
PPCA1_error=prediction_error(data, 1, 500, PPCA_2)
PPCA2_error=prediction_error(data, 2, 500, PPCA_2)
PPCA3_error=prediction_error(data, 3, 500, PPCA_2)
PPCA4_error=prediction_error(data, 4, 500, PPCA_2)
PPCA17_error=prediction_error(data, 17, 500, PPCA_2)

table = data.frame(
  a = c('Isotropic', 'Diagonal', 'PPCA', '', '', '', 'Full'),
  b = c(0, '(/)', 1, 2, 3, 4, '(17)'),
  c = c(1, 18, 19, 36, 52, 67, 171),
  d = c(Isotropic_error, Diagonal_error, PPCA1_error, PPCA2_error, PPCA3_error, PPCA4_error, PPCA17_error)
  )

names(table) = c('Covariance model', 'q(equivalent)', 'Number of parameters', 'Prediction error')

library('stargazer')
stargazer(table, summary=FALSE, rownames=FALSE, type = 'html')


