PPCA = function(t, q, epsilon=0.001, method = 'ML'){
  
#Input data:t
  t = as.matrix(t)
  
#t has the shape of d*N 
  d = dim(t)[1]
  N = dim(t)[2]
  
#Mu is the average of each rows
  Mu = rowSums(t) / N
  
#Compute the variance S
  S = matrix(data = 0, nrow = d, ncol = d)
  for (i in 1 : N){S = S + (t[, i] - Mu) %*% t(t[, i] - Mu)}
  S = S / N
  
#1.Use the result of the maximum likelihood directly
  if (method == 'ML'){
 
#Calculate the eigenvalue and eigenvector of S
  ev = eigen(S)
  
#Calculate sigma square
  if (q + 1 == d){sigma = sum(ev$values[d]) / (d - q)}
  if (q + 1 < d){sigma = sum(ev$values[(q + 1) : d]) / (d - q)}
  print(sigma)
  
#Calculate W
  U = ev$vectors[, 1 : q]
  if (q == 1){lambda = ev$values[1]}
  if (q > 1){lambda = diag(ev$values[1 : q])}
  W = U %*% sqrt(lambda - sigma * diag(q))
  print(W)
  
#Calculate the desired data X
  M = t(W) %*% W + sigma * diag(q)
  X = solve(M) %*% t(W) %*% (t-Mu)
  }

#2.Use EM algorithm
  if (method == 'EM'){
  
#Generate inital W and sigma_square 
   ev = eigen(S)
   U = ev$vectors[, 1 : q]
   if (q == 1){lambda = ev$values[1]}
   if (q > 1){lambda = diag(ev$values[1 : q])}
   
   W_old = U %*% sqrt(lambda)
   W_new = U %*% sqrt(lambda)
   sigma_old = 2
   sigma_new = 5
  
#EM algorithm
  while (sqrt(sum((W_old - W_new) ^ 2)) > epsilon || abs(sigma_new - sigma_old) > epsilon){
    M = t(W_new) %*% W_new + sigma_new * diag(q)
    W_old = W_new
    sigma_old = sigma_new
    W_new = S %*% W_old %*% solve(sigma_old * diag(q) + solve(M) %*% t(W_old) %*% S %*% W_old)
    sigma_new = sum(diag(S - S %*% W_old %*% solve(M) %*% t(W_new)))/d
    
    print(W_new)
    print(sigma_new)
  }
  
#Calculate the desired data X
  M = t(W_new) %*% W_new + sigma_new * diag(q)
  X = solve(M) %*% t(W_new) %*% (t - Mu)
  }
  
  return(X)
}






