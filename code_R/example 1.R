#Example: using tobamovirus data, and missing data experiment

library('limSolve')
path = "/code_R" 
setwd(path)  
source('PPCA.R')
source('PCA.R')

tobamovirus = read.csv('/data/tobamovirus.csv')
data = t(as.matrix(tobamovirus))

pca_data = t(PCA(data, 2))
plot(pca_data, col = 'white')
text(pca_data[, 1], pca_data[, 2], labels = c(1:38))

ppca_data = t(PPCA(data, 2, 0.001, 'EM'))
plot(ppca_data, col = 'white')
text(ppca_data[, 1], ppca_data[, 2], labels = c(1:38))


#########################

#EM algorithm for missing data
PPCA_missing = function(missing_data, q){
 
#Record the location of the missing data and fill NA with 0
  O = matrix(1, dim(missing_data)[1], dim(missing_data)[2])
  O[is.na(missing_data)] = 0
  Y = missing_data
  Y[is.na(missing_data)] = 0

#Initalization
  d = dim(missing_data)[1]
  N = dim(missing_data)[2]
  data_median = missing_data
  for(i in 1:d){data_median[i,][is.na(data_median[i,])] = median(missing_data[i,], na.rm = T)}
  
  m = rowMeans(data_median)
  S = matrix(data = 0, nrow = d, ncol = d)
  for (i in 1 : N){S = S + (data_median[, i] - m) %*% t(data_median[, i] - m)}
  S = S / N
  
  ev = eigen(S)
  if (q + 1 == d){v = sum(ev$values[d]) / (d - q)}
  if (q + 1 < d){v = sum(ev$values[(q + 1) : d]) / (d - q)}
  
  U = ev$vectors[, 1 : q]
  if (q == 1){lambda = ev$values[1]}
  if (q > 1){lambda = diag(ev$values[1 : q])}
  W = U %*% sqrt(lambda - diag(v, q))
  
  X = matrix(0, q, N)
  record = 1
  epsilon = 0.001
  
#Update parameters using EM algorithm
  while(abs(record-v) >= epsilon){
    
    record = v
    print(v)
#E step:
    for(j in 1 : N){
      W_j = W
      M_j = m
      
      for(i in 1 : d){
        if(O[i, j] == 0){
          W_j[i, ] = 0
          M_j[i] = 0 
          }
        }
      
      Psi = t(W_j) %*% W_j + diag(v, q, q)
      X[, j] = Solve(Psi) %*% t(W_j) %*% (Y[, j] - M_j)
      assign(paste('Sigma', j, sep = ''), v * Solve(Psi))
    }
   
#M step:
    for(i in 1 : d){
      X_i = X
      for(j in 1 : N){if(O[i, j] == 0){X_i[, j]=0}}
      
#Update m
      m[i] = 0
      for(j in 1 : N){
        if(O[i, j] == 1){m[i] = m[i] + Y[i, j] - W[i, ] %*% X[, j]}
      }
      m[i] = m[i]/sum(O[i, ])
      
#Update W
      W[i, ] = 0
      C = matrix(0, q, q)
      for(j in 1 : N){
        if(O[i, j] == 1){C = C + get(paste('Sigma', j, sep = ''))}
      }
      W[i, ] = t(Y[i, ] - m[i]) %*% t(X_i) %*% Solve(X_i %*% t(X_i) + C)
    }

#Update v    
    v = 0
    for (i in 1 : d){
      for (j in 1 : N){
        if (O[i, j] == 1){
          v = v + ((Y[i, j] - t(W[i, ]) %*% X[, j] - m[i])^2 + t(W[i, ]) %*% get(paste('Sigma', j, 
              sep = '')) %*% W[i, ]) / N
        }
      }
    }
    v = as.numeric(v)
  }
  
  return(X)
}

#Generate missing data
random_matrix = matrix(runif(18 * 38), nrow = 18, ncol = 38)
random_matrix[which(random_matrix > 0.2)] = 1
random_matrix[which(random_matrix <= 0.2)] = NA
missing_data = random_matrix * data

#Experiment
data_imputed = t(PPCA_missing(missing_data = missing_data, 2))
plot(data_imputed, col='white')
text(data_imputed[, 1], data_imputed[, 2], c(1:38))










