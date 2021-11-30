#Packages:

library("devtools")
library('roxygen2')
library("mathjaxr")
library('dplyr')
library('ggplot2')
library("knitr")
library('Rdpack')

#--------------------------------------------------------------------#

# Add function to R package which fits our model:

# All R functions in this R package belong in the R directory.


#--------------------------------------------------------------------#
#Generate function documentation .Rd file using roxygen syntax in model fit function:

setwd('C:/Users/baerd/Desktop/BayesianLMMFS')

devtools::document()


#--------------------------------------------------------------------#

#Install package:

setwd("..")
install("BayesianLMMFS")
library('BayesianLMMFS')

?BayesianLMMFS()


# uninstall('BayesianLMMFS')

#--------------------------------------------------------------------#

#Test to make sure function in package works:

set.seed(123)

#Define data settings:

n<-100
p<-16
K<-4
J<-3
p_k<-rep(p/K,K)
q<-2

#----------------------------------------------------------------------------------#

#Generate time-invariant feature data:

X<-matrix(rnorm(n*p),nrow=n,ncol=p)

#----------------------------------------------------------------------------------#

#Define true B:

Beta_k<-list()

for(k in 1:K){

  Beta_k[[k]]<-matrix(rnorm(J*p_k[k]),nrow=p_k[k],ncol=J)

}

#Introduce sparsity:

#Group-level feature sparsity:

Beta_k[[1]]<-0*Beta_k[[1]]

#Individual feature sparsity:

Beta_k[[2]][1,]<-0*Beta_k[[2]][1,]

#Collapse:

Beta<-dplyr::bind_rows(lapply(Beta_k,as.data.frame))%>%as.matrix

#vec(Beta):

beta<-matrix(Beta,nrow=J*p,ncol=1)

#----------------------------------------------------------------------------------#

#Define random effect feature data (time-varying):

Z<-list()

for(i in 1:n){

  Z[[i]]<-matrix(0,nrow=J,ncol=q)

  #For random intercept:

  Z[[i]][,1]<-1

  #For random slope:

  Z[[i]][,2]<-sort(rexp(n=J,rate=1/20))

  #Make first occasion 0:

  Z[[i]][,2][1]<-0

}

#----------------------------------------------------------------------------------#

#Define sigma2:

sigma2<-1/20

sigma2_I_J<-sigma2*diag(J)

#Define G:

G<-matrix(c(1/2,-10^-3,
            -10^-3,10^-4),
          q,q,byrow = T)

#----------------------------------------------------------------------------------#

#Generate outcome data:

Y_transpose<-matrix(0,nrow=J,ncol=n)

for (i in 1:n){

  #This is X_i (J x Jp):

  X_kron_i<-diag(J)%x%t(X[i,])

  #Mean structure:

  SS_MS_i<-X_kron_i%*%beta

  #Covariance structure:

  cov_y_i<-Z[[i]]%*%G%*%t(Z[[i]])+sigma2_I_J

  #Outcome data:

  Y_transpose[,i]<-MASS::mvrnorm(n=1,
                                 mu=SS_MS_i,
                                 Sigma=cov_y_i)

}

#----------------------------------------------------------------------------------#

#Fit model to simulated data:

I_J<-diag(J)

I_q<-diag(q)

model_results<-

  BayesianLMMFS::BayesianLMMFS(

    Y_transpose,
    X,
    Z,
    p_k,

    nsim=200,
    burn=100,
    thin=1,

    Q=10^-3*I_J,

    C_0=10^-3*I_q,

    raw_MCMC_output = T
  )



#Inspect feature parameter estimates:

model_results$MCMC_summary%>%head

model_results$MCMC_output%>%head



model_results$MCMC_output$subject%>%class




model_results$subject$q%>%class

apply(model_results$MCMC_output,2,class)
model_results$MCMC_output$parameter%>%unique
