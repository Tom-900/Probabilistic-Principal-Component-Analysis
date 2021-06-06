#Example about mixtures of PPCA models

library('limSolve')
path = "/code_R" 
setwd(path)  
source('PPCA.R')
source('PCA.R')

tobamovirus = read.csv('D:/statistics/CUHK_PPCA/data/tobamovirus.csv')
data = t(as.matrix(tobamovirus))
t = data

d = dim(data)[1]
N = dim(data)[2]
q = 2
K = 3

# Initalization
Pi = array(0, K)
record = array(1, K)
Mu = matrix(0, d, K)
sigma = array(0, K)
P = matrix(0, N, K)

R = matrix(runif(N*K), N, K)
sum = rowSums(R)
R = R/sum

epsilon = 0.001
flag = 0

while(sqrt(sum((record - Pi)^2)) >= epsilon){
#Record the number of loop  
  flag = flag + 1
  
  for (k in 1 : K){
#Record the last Pi
    record[k] = Pi[k]
        
#Update Pi, Mu
    Pi[k] = sum(R[, k])/ N
    Mu[, k] = (t %*% R[, k])/sum(R[, k])
    
#Update S
    S = matrix(0, d, d)
    for (n in 1 : N){S =  S + R[n, k] * (t[, n] - Mu[, k]) %*% t(t[, n] - Mu[, k])}
    S =  S/(Pi[k] * N)

#Compute new value of U and lambda according to the eigenvalue and eigenvector of S
#Update sigma
    ev = eigen(S)
    if (q + 1 == d){sigma[k] = sum(ev$values[d]) / (d - q)}
    if (q + 1 < d){sigma[k] = sum(ev$values[(q + 1) : d]) / (d - q)}
    U = ev$vectors[, 1 : q]
    
    if (q == 1){lambda = ev$values[1]}
    if (q > 1){lambda = ev$values[1 : q]}

#Update W        
    assign(paste('W', k, sep = ''), U %*% sqrt(lambda - sigma[k] * diag(q)))

#Compute C by sigma and W
    assign(paste('C', k, sep=''), sigma[k]*diag(d) + get(paste('W', k, sep='')) %*% 
    t(get(paste('W', k, sep=''))))

#Compute the conditional probability of t to k    
    for (n in 1 : N){P[n, k] = (sqrt(det(get(paste('C', k, sep=''))))^(-1)) * exp(-(t(t[, n]-Mu[, k])%*%
    Solve(get(paste('C', k, sep='')))%*%(t[, n]-Mu[, k]))/2)}
  }

#Update R[n, k] 
  for (n in 1 : N){
    for (k in 1 : K){
      R[n, k] = P[n, k]*Pi[k]/(P[n, ]%*%Pi)
    }
  }
}

print(Pi)
rm(ev, P, S, U, lambda, record, sum)

t1 = t[, which(apply(R, 1, which.max) == 1)]
M1 = t(W1) %*% W1 + sigma[1]*diag(q)
x1 = solve(M1) %*% t(W1) %*% (t1-Mu[, 1])
plot(t(x1), col = 'white')
text(t(x1)[, 1], t(x1)[, 2], labels = which(apply(R, 1, which.max) == 1))

#The data No.8 is obviously a outlier of this cluster 
t2 = t[, which(apply(R, 1, which.max) == 2)]  
M2 = t(W2) %*% W2 + sigma[2]*diag(q)
x2 = solve(M2) %*% t(W2) %*% (t2-Mu[, 2])
plot(t(x2[,-6]), col = 'white')
text(t(x2)[,-6][, 1], t(x2)[,-6][, 2], labels = which(apply(R, 1, which.max) == 2)[-6])  

t3 = t[, which(apply(R, 1, which.max) == 3)]  
M3 = t(W3) %*% W3 + sigma[3]*diag(q)
x3 = solve(M3) %*% t(W3) %*% (t3-Mu[, 3])
plot(t(x3), col = 'white')
text(t(x3)[, 1], t(x3)[, 2], labels = which(apply(R, 1, which.max) == 3))   
  
  
  
  
  
  
  



