PCA = function(t, q){
  
  d = dim(t)[1]
  N = dim(t)[2]
  Mu = rowSums(t) / N
  S = matrix(data = 0, nrow = d, ncol = d)
  for (i in 1 : N){S = S + ((t[, i] - Mu) %*% t(t[, i] - Mu))}
  S = S / N
  
  ev = eigen(S)
  W = ev$vectors[, c(1 : q)]
  X = t(W) %*% (t - Mu)
  
  return(X)
}