
#' Fit the Bayesian linear mixed model for bi-level feature selection (BayesianLMMFS) described in Baer et al. (2021)
#'
#' This function implements a Markov chain Monte Carlo (MCMC) Gibbs sampling algorithm to fit
#' a Bayesian linear mixed model for bi-level feature selection;
#' this model is developed for the data scenario where we have time-invariant feature data and
#' possibly irregularly-spaced longitudinal outcome data;
#' this model specifies time-varying
#' feature parameters, and allows for the incorporation of group structure into
#' the feature selection process.
#' \loadmathjax
#'
#' @importFrom Rdpack reprompt
#'
#' @param Y_transpose A \mjeqn{J \times n}{}
#' matrix of longitudinal outcome data,
#' where \mjeqn{J}{J}
#'  is the number of measurement occasions common to all subjects,
#' and \mjeqn{n}{n} is the number of subjects.
#' This model assumes multivariate normality of these outcome data.
#'
#' @param X An \mjeqn{n \times p}{}
#' matrix of time-invariant feature data, where
#' \mjeqn{p}{p} is the number of features.
#'
#'
#' @param Z A list consisting of
#' \mjeqn{n \underset{J\times q}{\mathbf{Z}_i}}{}
#' matrices, where each
#' \mjeqn{\underset{J\times q}{\mathbf{Z}_i}}{}
#' is the matrix
#' of (time-varying) random effect feature data
#' corresponding to the random effect parameters,
#' \mjeqn{\underset{q\times 1}{\mathop{{{\mathbf{b}}_{i}}}}\,}{}
#' for
#' \mjeqn{i = 1, \cdots, n}{}.
#'
#' @param p_k A
#' \mjeqn{K \times 1}{}
#' matrix of the feature group sizes,
#' \mjeqn{p_k}{},
#' where
#' \mjeqn{K}{K}
#' is the number of feature groups.
#' Note that
#' \mjeqn{\sum\limits_{k=1}^{K}{{{p}_{k}}}=p}{}.
#'
#'
#' @param nsim The total number of MCMC
#' iterations to be performed.
#'
#' @param burn The number of MCMC
#' iterations to be discarded as part of the burn-in period.
#'
#' @param thin The thinning interval applied to the MCMC
#' sample; this is useful if the user wishes to conserve computational
#' memory when fitting the model.
#'
#' @param mcmc_div The interval at which the current MCMC
#' iteration number will be reported; the
#' default value is
#' \code{floor((nsim-burn)/4)}.
#'
#'
#'
#' @param d The degrees of freedom parameter for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{J\times J}{\mathbf{\Sigma}}}{};
#' note that
#' \mjeqn{\underset{J\times J}{\mathbf{\Sigma}}}{}
#' represents the
#' covariance for each
#' of the
#'\mjeqn{l = 1, \cdots, p}{}
#' time-varying feature parameters
#' within the matrix
#' \mjeqn{\underset{p\times J}{\mathbf{B}}}{};
#' we recommend setting \code{d} to the smallest possible
#' value, which is
#' \mjeqn{J}{J};
#' the default value is \mjeqn{J}{J}.
#'
#' @param Q The
#' \mjeqn{J\times J}{}
#' scale matrix parameter for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{J\times J}{\mathbf{\Sigma}}}{}.
#' Note that
#' \code{Q}
#' must be positive-definite.
#'
#' @param nu_0 The degrees of freedom parameter for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{q\times q}{\mathbf{G}}}{};
#' note that
#' \mjeqn{\underset{q\times q}{\mathbf{G}}}{}
#' represents the
#' covariance for each
#' of the
#' vectors of random effect parameters,
#' \mjeqn{\underset{q\times 1}{\mathbf{b}_i}}{}.
#' We recommend setting \code{nu_0} to the smallest possible
#' value, which is
#' \mjeqn{q}{q};
#' the default value is \mjeqn{q}{q}.
#'
#' @param C_0 The
#' \mjeqn{q\times q}{}
#' scale matrix parameter for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{q\times q}{\mathbf{G}}}{}.
#' Note that
#' \code{C_0}
#' must be positive-definite.
#'
#' @param alpha The shape parameter for
#' the inverse-gamma-distributed
#' \mjeqn{\sigma^2}{};
#' note that \mjeqn{\sigma^2}{}
#' represents the
#' (conditional)
#' variance for
#' each vector of longitudinal outcome data,
#' \mjeqn{\underset{J\times 1}{\mathbf{y}_i}}{},
#' for
#' \mjeqn{i = 1, \cdots, n}{}.
#' That is,
#' \mjeqn{\operatorname{cov}\left( \underset{J\times 1}{\mathbf{y}_i} \right) =
#' {{\sigma }^{2}}\underset{J\times J}{\mathop{{{\mathbf{I}}_{J}}}}\,}{}
#' \mjeqn{}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param gamma The scale parameter for
#' the inverse-gamma-distributed
#' \mjeqn{\sigma^2}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param o The shape parameter for
#' the inverse-gamma-distributed
#' \mjeqn{s^2}{};
#' note that \mjeqn{s^2}{}
#' represents the
#' variance
#' of the
#' spike-and-slab feature selection parameter,
#' \mjeqn{\tau_{l}^2}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param u The scale parameter for
#' the inverse-gamma-distributed
#' \mjeqn{s^2}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param a_1 The first shape parameter for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{};
#' note that
#' \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{}
#' represents the
#' probability
#' of the
#' group-level feature selection indicator variable,
#' \mjeqn{\pi_{0k}}{},
#' equaling
#' \mjeqn{1}{}.
#' The default value is \mjeqn{1}{1}.
#'
#' @param b_1 The second shape parameter for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{}.
#' The default value is \mjeqn{1}{1}.
#'
#' @param g The first shape parameter for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{{\tau }^{2}}}}}{};
#' note that
#' \mjeqn{{{\theta }_{{{\tau }^{2}}}}}{}
#' represents the
#' probability
#' of the
#' individual feature selection indicator variable,
#' \mjeqn{\pi_{0l}}{},
#' equaling
#' \mjeqn{1}{}.
#' The default value is \mjeqn{1}{1}.
#'
#' @param h The second shape parameter for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{{\tau }^{2}}}}}{}.
#' The default value is \mjeqn{1}{}.
#'
#' @param raw_MCMC_output If set to \code{TRUE}, the output for \code{BayesianLMMFS} will include a data frame
#' containing un-summarized MCMC output.
#' The default value is \code{TRUE}.
#'
#' @return If \code{raw_MCMC_output=TRUE}, \code{BayesianLMMFS} returns a list with two data frames named
#' \code{"MCMC_output"} and \code{"MCMC_summary"};
#' \code{"MCMC_output"} is a data frame with the un-summarized MCMC output, while
#' \code{"MCMC_summary"} is a data frame with the summarized MCMC output.
#' If \code{raw_MCMC_output=FALSE}, \code{BayesianLMMFS} returns a list with only one data frame named
#' \code{"MCMC_summary"}
#'
#' The \code{"MCMC_output"} data frame containing the un-summarized MCMC output has the following columns:
#' \itemize{
#'  \item{"feature"}{} A number denoting the \mjeqn{l=1,\cdots,p}{} feature parameters.
#'
#'  \item{"occasion"}{} A character denoting the \mjeqn{j=1,\cdots,J}{} occasions; each occasion
#'  is denoted by the character \code{"tj"}.
#'
#'  \item{"feature_group"}{} A number denoting the \mjeqn{k=1,\cdots,K}{} feature groups.
#'
#'  \item{"subject"}{} A character denoting the \mjeqn{i=1,\cdots,n}{} subjects.
#'
#'  \item{"parameter"}{} A character denoting the model parameter; the model parameters include:
#'
#'  \itemize{
#'
#'  \item{"b"}{} the random effect parameters,
#'  \mjeqn{\underset{q\times 1}{\mathop{{{\mathbf{b}}_{i}}}}\,}{}.
#'
#'  \item{"beta"}{} The feature parameters,
#'  \mjeqn{\underset{p\times J}{\mathop{\mathbf{B}}}\,}{}.
#'
#'  \item{"G"}{} The random effect parameter covariance matrix,
#'  \mjeqn{\underset{q\times q}{\mathop{\mathbf{G}}}\,}{}.
#'
#'  \item{"pi_0k"}{} The group-level feature selection indicator parameters,
#'  \mjeqn{{{\pi}_{0k}}}{}.
#'
#'  \item{"pi_0l"}{} The individual feature selection indicator parameters,
#'  \mjeqn{{{\pi}_{0l}}}{}.
#'
#'  \item{"s2"}{} The variance parameter of
#'  \mjeqn{\tau _{l}^{2}}{},
#'  \mjeqn{s^{2}}{}.
#'
#'  \item{"Sigma"}{} The feature parameter covariance matrix,
#'  \mjeqn{\underset{J\times J}{\mathop{\mathbf{\Sigma }}}\,}{}.
#'
#'  \item{"sigma2"}{} The variance parameter of
#'  \mjeqn{\underset{J\times 1}{\mathbf{y}_i}}{},
#'  \mjeqn{\sigma^{2}}{}.
#'
#'  \item{"tau_l2"}{} The slab parameters,
#'  \mjeqn{\tau _{l}^{2}}{}.
#'
#'  \item{"theta_beta_tilde"}{} The probability parameter for group-level feature selection,
#'  \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{}.
#'
#'  \item{"theta_tau2"}{} The probability parameter for individual feature selection,
#'  \mjeqn{{{\theta }_{\tau^{2}}}}{}.
#'
#'}
#'
#'  \item{"row_occasion"}{} A character denoting the \mjeqn{J^2}{} elements of
#'  the feature parameter covariance matrix,
#'  \mjeqn{\underset{J\times J}{\mathop{\mathbf{\Sigma }}}\,}{};
#'  each element
#'  is denoted by the character \code{"tj"}.
#'
#'  \item{"q"}{} A character denoting the \mjeqn{q}{} random effect parameters,
#'  \mjeqn{\underset{q\times 1}{\mathop{{{\mathbf{b}}_{i}}}}\,}{}.
#'
#'  \item{"row_q"}{} A character denoting the \mjeqn{q^2}{} elements of the
#'  random effect parameter covariance matrix,
#'  \mjeqn{\underset{q\times q}{\mathop{\mathbf{G}}}\,}{}.
#'
#'  \item{"comp_time"}{} The elapsed computation time in seconds.
#'
#'  \item{"mcmc_iter"}{} The MCMC iteration number.
#'
#'  \item{"value"}{} The numeric posterior estimate of the model parameter at each MCMC iteration.
#'
#' }
#'
#' The \code{"MCMC_summary"} data frame containing the summarized MCMC output has the following columns:
#' \itemize{
#'
#'  \item{"feature"}{} A character denoting the \mjeqn{l=1,\cdots,p}{} feature parameters.
#'
#'  \item{"occasion"}{} A character denoting the \mjeqn{j=1,\cdots,J}{} occasions; each occasion
#'  is denoted by the character string \code{"tj"}.
#'
#'  \item{"feature_group"}{} A number denoting the \mjeqn{k=1,\cdots,K}{} feature groups.
#'
#'  \item{"subject"}{} A character denoting the \mjeqn{i=1,\cdots,n}{} subjects.
#'
#'  \item{"parameter"}{} A character denoting the model parameter; the model parameters are the
#'  same here as for the \code{"MCMC_output"} data frame.
#'
#'  \item{"row_occasion"}{} A character denoting the \mjeqn{J^2}{} elements of
#'  the feature parameter covariance matrix,
#'  \mjeqn{\underset{J\times J}{\mathop{\mathbf{\Sigma }}}\,}{};
#'  each element
#'  is denoted by the character \code{"tj"}.
#'
#'  \item{"q"}{} A character denoting the \mjeqn{q}{} random effect parameters.
#'
#'  \item{"row_q"}{} A character denoting the \mjeqn{q^2}{} elements of the
#'  random effect parameter covariance matrix,
#'  \mjeqn{\underset{q\times q}{\mathop{\mathbf{G}}}\,}{}.
#'
#'  \item{"MCMC_mean"}{} The numeric posterior mean for each model parameter.
#'
#'  \item{"MCMC_median"}{} The numeric posterior median for each model parameter.
#'
#'  \item{"MCMC_sd"}{} The numeric posterior standard deviation for each model parameter.
#'
#'  \item{"MCMC_LCL"}{} The numeric posterior \mjeqn{2.5^{th}}{} percentile for each model parameter;
#'  used to construct the \mjeqn{95}{} percent credible interval (CI) for each model parameter.
#'
#'  \item{"MCMC_UCL"}{} The numeric MCMC \mjeqn{97.5^{th}}{} percentile for each model parameter;
#'  used to construct the \mjeqn{95}{} percent CI for each model parameter.
#'
#'  \item{"comp_time"}{} The elapsed computation time in seconds.
#'
#' }
#'
#' @examples
#'
#' set.seed(123)
#'
#' #Define data settings:
#'
#' n<-100
#' p<-16
#' K<-4
#' J<-3
#' p_k<-rep(p/K,K)
#' q<-2
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Generate time-invariant feature data:
#'
#' X<-matrix(rnorm(n*p),nrow=n,ncol=p)
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Define true B:
#'
#' Beta_k<-list()
#'
#' for(k in 1:K){
#'
#'  Beta_k[[k]]<-matrix(rnorm(J*p_k[k]),nrow=p_k[k],ncol=J)
#'
#' }
#'
#' #Introduce sparsity:
#'
#' #Group-level feature sparsity:
#'
#' Beta_k[[1]]<-0*Beta_k[[1]]
#'
#' #Individual feature sparsity:
#'
#' Beta_k[[2]][1,]<-0*Beta_k[[2]][1,]
#'
#' #Collapse:
#'
#' Beta<-dplyr::bind_rows(lapply(Beta_k,as.data.frame))%>%as.matrix
#'
#' #vec(Beta):
#'
#' beta<-matrix(Beta,nrow=J*p,ncol=1)
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Define random effect feature data (time-varying):
#'
#' Z<-list()
#'
#' for(i in 1:n){
#'
#'   Z[[i]]<-matrix(0,nrow=J,ncol=q)
#'
#'   #For random intercept:
#'
#'   Z[[i]][,1]<-1
#'
#'   #For random slope:
#'
#'   Z[[i]][,2]<-sort(rexp(n=J,rate=1/20))
#'
#'   #Make first occasion 0:
#'
#'   Z[[i]][,2][1]<-0
#'
#' }
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Define sigma2:
#'
#' sigma2<-1/20
#'
#' sigma2_I_J<-sigma2*diag(J)
#'
#' #Define G:
#'
#' G<-matrix(c(1/2,-10^-3,
#'             -10^-3,10^-4),
#'             q,q,byrow = T)
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Generate outcome data:
#'
#' Y_transpose<-matrix(0,nrow=J,ncol=n)
#'
#' for (i in 1:n){
#'
#'   #This is X_i (J x Jp):
#'
#'   X_kron_i<-diag(J)%x%t(X[i,])
#'
#'   #Marginal mean structure:
#'
#'   SS_MS_i<-X_kron_i%*%beta
#'
#'   #Marginal covariance structure:
#'
#'   cov_y_i<-Z[[i]]%*%G%*%t(Z[[i]])+sigma2_I_J
#'
#'   #Longitudinal outcome data:
#'
#'   Y_transpose[,i]<-MASS::mvrnorm(n=1,
#'                      mu=SS_MS_i,
#'                      Sigma=cov_y_i)
#'
#' }
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Fit model to simulated data:
#'
#' I_J<-diag(J)
#'
#' I_q<-diag(q)
#'
#' model_results<-
#'
#' BayesianLMMFS(
#'
#'     Y_transpose,
#'     X,
#'     Z,
#'     p_k,
#'
#'     nsim=2000,
#'     burn=1000,
#'     thin=1,
#'
#'     Q=10^-3*I_J,
#'
#'     C_0=10^-3*I_q
#'  )
#'
#' #Inspect feature parameter estimates:
#'
#' model_results$MCMC_summary%>%dplyr::filter(parameter=='beta')

#----------------------------------------------------------------------------------#

#' @import dplyr
#' @importFrom MASS mvrnorm
#' @import MCMCpack
#' @import truncnorm
#' @import tidyr

#' @export BayesianLMMFS

BayesianLMMFS<-function(

  Y_transpose,

  X,

  Z,

  p_k,


  #-------#

  nsim,

  burn,

  thin,

  mcmc_div=floor((nsim-burn)/4),

  #-------#

  d=nrow(Y_transpose),

  Q,

  nu_0=ncol(Z[[1]]),

  C_0,

  alpha=10^-3,

  gamma=10^-3,

  o=10^-3,

  u=10^-3,

  a_1=1,

  b_1=1,

  g=1,

  h=1,

  #-------#

  raw_MCMC_output=T){

  #----------------------------------------------------------------------------------#

  #Organize observed/simulated data in order to fit MCMC algorithm:####

  #Computational time:

  start_comp_time<-Sys.time()

  #----------------------------------------------------------------------------------#

  #Define J and n:

  J<-nrow(Y_transpose)

  n<-ncol(Y_transpose)

  #Define diag(J)

  I_J<-diag(J)

  #--------------------------#

  #Create vec(Y^T) (Jn x 1):

  vec_Y_transpose<-matrix(Y_transpose,
                          nrow=(J*n),
                          ncol=1)


  #----------------------------------------------------------------------------------#

  #Define K:

  K<-length(p_k)

  #----------------------------------------------------------------------------------#

  #X (n x p):

  #Define p:

  p<-ncol(X)

  #----------------------------------------------------------------------------------#

  #Create fn for creating data matrices, whose
  #elements are I_J kron c_i^T:

  ss_data_matrix_fn<-function(data_matrix,
                              J){

    #Define diag(J)

    I_J<-diag(J)

    #--------------------------#

    data_matrix_list<-list()

    #--------------------------#

    #Define n:

    n<-nrow(data_matrix)

    #--------------------------#

    #Define p:

    p<-ncol(data_matrix)

    #--------------------------#

    for(i in 1:n){

      c_i_T<-
        data_matrix[i,]%>%
        matrix(.,
               nrow=1,
               ncol=p)

      data_matrix_list[[i]]<-
        I_J%x%c_i_T

    }


    data_matrix_list<-lapply(data_matrix_list,as.data.frame)

    #Collapse into a single df:

    data_matrix<-
      dplyr::bind_rows(data_matrix_list)%>%
      as.matrix

    return(data_matrix)

  }


  #--------------------------#

  #Define X_kron (nJ*Jp):

  X_kron_mat<-
    ss_data_matrix_fn(data_matrix=X,
                      J=J)

  #--------------------------#

  #For group level model only:

  #Creating x_ik (1 x p_k) and x_i(-k) (1 x p_(-k))

  #Creating X_k (nJ x Jp_k)
  #and X_-k (nJ x Jp_-k)

  X_dim_df<-data.frame(n=n,
                       p_k=p_k,
                       k=1:K)

  names(X_dim_df)<-c('n',
                     'p_k',
                     'k')

  X_dim_df<-X_dim_df%>%
    dplyr::mutate(cum_p_k=cumsum(p_k),
                  lag_cum_p_k=lag(cum_p_k)+1)

  X_dim_df$lag_cum_p_k[1]<-1


  #--------------------------#

  #Create fn for creating group-level data matrices

  GROUP_data_matrix_fn<-function(data_matrix,
                                 J,
                                 X_dim_df){

    #Define diag(J)

    I_J<-diag(J)

    #--------------------------#

    #Define n:

    n<-nrow(data_matrix)

    #--------------------------#

    #Define p:

    p<-ncol(data_matrix)

    #--------------------------#

    #Define k:

    K<-nrow(X_dim_df)

    X_k<-list()

    X_NOT_k<-list()

    #--------------------------#

    for(i in 1:n){

      X_k[[i]]<-list()

      X_NOT_k[[i]]<-list()

      #--------------------------#


      for(k in 1:K){

        k_select<-
          X_dim_df[k,]

        p_select<-
          k_select$lag_cum_p_k:k_select$cum_p_k

        p_NOT_select<-
          which(((1:p)%in%p_select)==F)

        #--------------------------#

        #X_ik (J X Jp_k)

        #This is 1 x p_k:

        X_ik_T<-
          data_matrix[i,p_select]%>%
          matrix(.,
                 nrow=1,
                 ncol=p_k[k])

        X_k[[i]][[k]]<-
          I_J%x%X_ik_T

        #--------------------------#

        #X_i_NOT_k (J x Jp_NOT_k)

        #This is 1 x p_-k:

        X_i_NOT_k_T<-
          data_matrix[i,p_NOT_select]%>%
          matrix(.,
                 nrow=1,
                 ncol=p-p_k[k])

        X_NOT_k[[i]][[k]]<-
          I_J%x%X_i_NOT_k_T

        #--------------------------#


      }#end k loop

      names(X_k[[i]])<-
        paste('k',1:K,sep='_')

      names(X_NOT_k[[i]])<-
        paste('k',1:K,sep='_')

      #--------------------------#

    }#end i loop

    names(X_k)<-paste('n',1:n,sep='_')

    names(X_NOT_k)<-paste('n',1:n,sep='_')


    #--------------------------#


    X_STAR_k<-list()

    X_STAR_NOT_k<-list()

    #--------------------------#

    for(k in 1:K){

      X_STAR_k[[k]]<-
        lapply(X_k,"[", k)

      X_STAR_NOT_k[[k]]<-
        lapply(X_NOT_k,"[", k)

      #----------------------------------------------------------#


      X_STAR_k[[k]]<-
        lapply(X_STAR_k[[k]],as.data.frame)

      X_STAR_NOT_k[[k]]<-
        lapply(X_STAR_NOT_k[[k]],as.data.frame)

      #----------------------------------------------------------#

      #Collapse:

      #X_k (nJ x Jp_k)

      X_STAR_k[[k]]<-
        dplyr::bind_rows(X_STAR_k[[k]])

      X_STAR_k[[k]]<-
        as.matrix(X_STAR_k[[k]])

      #--------------------------#


      #X_NOT_k (nJ x Jp_NOT_k)

      X_STAR_NOT_k[[k]]<-
        dplyr::bind_rows(X_STAR_NOT_k[[k]])

      X_STAR_NOT_k[[k]]<-
        as.matrix(X_STAR_NOT_k[[k]])

      #--------------------------#



    }#end k

    names(X_STAR_k)<-paste('k',1:K,sep='_')

    names(X_STAR_NOT_k)<-paste('k',1:K,sep='_')

    #--------------------------#

    return(list(
      'X_STAR_k'=X_STAR_k,
      'X_STAR_NOT_k'=X_STAR_NOT_k
    ))

  }#end fn


  #--------------------------#

  X_k_SS<-GROUP_data_matrix_fn(data_matrix=X,
                               J=J,
                               X_dim_df = X_dim_df)

  X_STAR_k<-X_k_SS$X_STAR_k


  X_STAR_NOT_k<-X_k_SS$X_STAR_NOT_k



  #--------------------------#


  #For just bi-level feature selection:

  #Creating x_il (1 x 1) and x_i(-l) (1 x (p-1))

  #Creating X_l (nJ x J)
  #and X_-l (nJ x J(p-1))

  #--------------------------#


  #Create fn for creating bi-level data matrices

  BI_data_matrix_fn<-function(data_matrix,
                              J){

    #Define diag(J)

    I_J<-diag(J)

    #--------------------------#

    #Define n:

    n<-nrow(data_matrix)

    #--------------------------#

    #Define p:

    p<-ncol(data_matrix)

    #--------------------------#


    X_l<-list()

    X_NOT_l<-list()

    #--------------------------#

    for(i in 1:n){

      X_l[[i]]<-list()

      X_NOT_l[[i]]<-list()

      #--------------------------#

      for(l in 1:p){

        p_select<-l

        p_NOT_select<-which(((1:p)%in%p_select)==F)

        #--------------------------#

        #X_il (J X J)

        #This is x_il (1 x 1)

        x_il<-
          data_matrix[i,l]%>%
          matrix(.,
                 nrow=1,
                 ncol=1)

        X_l[[i]][[l]]<-
          I_J%x%x_il

        #--------------------------#

        #X_i_NOT_l (J x J(p-1))

        #This is x_i_NOT_l (1 x (p-1))

        x_i_NOT_l_T<-
          data_matrix[i,p_NOT_select]%>%
          matrix(.,
                 nrow=1,
                 ncol=p-1)

        X_NOT_l[[i]][[l]]<-
          I_J%x%x_i_NOT_l_T

        #--------------------------#

      }#end l loop

      names(X_l[[i]])<-
        paste('l',1:p,sep='_')

      names(X_NOT_l[[i]])<-
        paste('l',1:p,sep='_')

    }#end i loop

    names(X_l)<-
      paste('n',1:n,sep='_')

    names(X_NOT_l)<-
      paste('n',1:n,sep='_')


    #--------------------------#


    #For the bi-level selection model only:

    X_STAR_l<-list()

    X_STAR_NOT_l<-list()

    #--------------------------#


    for(l in 1:p){

      X_STAR_l[[l]]<-lapply(X_l,"[",l)

      X_STAR_NOT_l[[l]]<-lapply(X_NOT_l,"[",l)

      #----------------------------------------------------------#

      X_STAR_l[[l]]<-lapply(X_STAR_l[[l]],as.data.frame)

      X_STAR_NOT_l[[l]]<-lapply(X_STAR_NOT_l[[l]],as.data.frame)

      #----------------------------------------------------------#

      #Collapse:

      #X_l (nJ x J)

      X_STAR_l[[l]]<-
        dplyr::bind_rows(X_STAR_l[[l]])

      X_STAR_l[[l]]<-
        as.matrix(X_STAR_l[[l]])

      #--------------------------#

      #X_NOT_l (nJ x J(p-1))

      X_STAR_NOT_l[[l]]<-
        dplyr::bind_rows(X_STAR_NOT_l[[l]])

      X_STAR_NOT_l[[l]]<-
        as.matrix(X_STAR_NOT_l[[l]])


    }#end l loop

    names(X_STAR_l)<-paste('l',1:p,sep='_')

    names(X_STAR_NOT_l)<-paste('l',1:p,sep='_')


    #----------------------------------------------------------#

    return(list(
      'X_STAR_l'=X_STAR_l,
      'X_STAR_NOT_l'=X_STAR_NOT_l
    ))

  }#end fn


  #--------------------------#


  X_l_SS<-
    BI_data_matrix_fn(data_matrix=X,
                      J=J)

  X_STAR_l<-X_l_SS$X_STAR_l


  X_STAR_NOT_l<-X_l_SS$X_STAR_NOT_l



  #----------------------------------------------------------------------------------#

  #Create fn for creating Z (nJ x nq):

  Z_data_matrix_fn<-function(data_list,
                             J,
                             q){

    #Define diag(J)

    I_J<-diag(J)

    #--------------------------#

    #Define n:

    n<-length(data_list)

    #--------------------------#

    #Z_i (J x q)

    #Number of REs:

    q<-q

    Z_mat<-matrix(0,
                  nrow=n*J,
                  ncol=n*q)

    Z_dim_df<-data.frame(i=1:n,
                         J=J,
                         q=q)

    Z_dim_df<-Z_dim_df%>%
      dplyr::mutate(cum_J=cumsum(J),
                    cum_q=cumsum(q),
                    lag_cum_J=lag(cum_J)+1,
                    lag_cum_q=lag(cum_q)+1)

    Z_dim_df$lag_cum_J[1]<-1

    Z_dim_df$lag_cum_q[1]<-1

    #--------------------------#

    for(i in 1:n){

      #Z (nJ x nq)

      Z_mat[Z_dim_df$lag_cum_J[i]:Z_dim_df$cum_J[i],
            Z_dim_df$lag_cum_q[i]:Z_dim_df$cum_q[i]]<-data_list[[i]]

    }#end i

    #--------------------------#

    return(Z_mat)

  }#end fun.

  #--------------------------#

  #Define q:

  q<-ncol(Z[[1]])

  #--------------------------#

  Z_mat<-
    Z_data_matrix_fn(data_list=Z,
                     J=J,
                     q=q)


  #----------------------------------------------------------------------------------#


  #################################

  #Start of MCMC function:####

  #################################

  #----------------------------------------------------------------------------------#

  #Initial values for stochastic model parameters:####

  #----------------------------------------------------------------------------------#

  #Model parameter initialization:

  print('Creating data objects re: initial MCMC values')

  #Beta_tilde (p x J):####

  #N.b. our Gibbs sampler only updates beta_tilde_k (p_k*J x 1 ),

  #This is p x J:

  Beta_tilde<-matrix(0,
                     nrow=p,
                     ncol=J)

  #This is p_k x J:

  Beta_tilde_k<-list()

  #This is vec(Beta_tilde_k) which is p_kJ x 1

  beta_tilde_k<-list()


  for(k in 1:K){

    k_select<-X_dim_df[k,]

    p_select<-k_select$lag_cum_p_k:k_select$cum_p_k


    #This is p_k x J

    Beta_tilde_k[[k]]<-
      Beta_tilde[p_select,]%>%
      matrix(.,
             nrow=length(p_select),
             ncol=J)


    #This is vec(Beta_k) which is p_kJ x 1

    beta_tilde_k[[k]]<-
      matrix(Beta_tilde_k[[k]],
             nrow=length(p_select)*J,
             ncol=1)



  }

  names(beta_tilde_k)<-
    names(Beta_tilde_k)<-
    paste('k',1:K,sep='_')


  #----------------------------------------------------------------------------------#

  #For storing Beta_k (p_k x J):####

  Beta_k<-list()

  #-------------------------#

  #For storing beta_k (p_kJ x 1):####

  #We need this for outputting MCMC samples for Beta (p x J):

  beta_k<-list()


  #----------------------------------------------------------------------------------#

  #beta_tilde_l^T (1 x J):####

  #This is beta_tilde_l^T (1 x J):

  beta_tilde_l<-list()

  for(l in 1:p){

    p_select<-l

    beta_tilde_l[[l]]<-Beta_tilde[p_select,]%>%
      matrix(.,
             nrow=J,
             ncol=1)

  }

  names(beta_tilde_l)<-paste('l',1:p,sep='_')

  #----------------------------------------------------------------------------------#


  #Beta_NOT_l (p-1 x J):####

  Beta<-matrix(0,
               nrow=p,
               ncol=J)


  #This is beta_NOT_l ((p-1)J x 1):

  beta_NOT_l<-list()

  for(l in 1:p){

    p_select<-l

    p_NOT_select<-which(((1:p)%in%p_select)==F)

    Beta_NOT_l<-Beta[p_NOT_select,]%>%
      matrix(.,
             nrow=length(p_NOT_select),
             ncol=J)


    beta_NOT_l[[l]]<-matrix(Beta_NOT_l,
                            nrow=length(p_NOT_select)*J,
                            ncol=1)

  }

  names(beta_NOT_l)<-paste('l',1:p,sep='_')

  #----------------------------------------------------------------------------------#

  #Beta_NOT_k (p(-k) x J):####

  #This is beta_NOT_k (p(-k)J x 1):

  beta_NOT_k<-list()

  for(k in 1:K){

    k_select<-X_dim_df[k,]

    p_select<-k_select$lag_cum_p_k:k_select$cum_p_k

    p_NOT_select<-which(((1:p)%in%p_select)==F)

    Beta_NOT_k<-Beta[p_NOT_select,]%>%
      matrix(.,
             nrow=length(p_NOT_select),
             ncol=J)

    beta_NOT_k[[k]]<-matrix(Beta_NOT_k,
                            nrow=length(p_NOT_select)*J,
                            ncol=1)

  }

  names(beta_NOT_k)<-paste('k',1:K,sep='_')


  #----------------------------------------------------------------------------------#

  #pi_0k:####

  pi_0k<-matrix(1,
                nrow=K,
                ncol=1)


  #----------------------------------------------------------------------------------#


  #pi_0l:####

  pi_0l<-matrix(1,
                nrow=p,
                ncol=1)

  #----------------------------------------------------------------------------------#

  #theta_beta_tilde:####

  theta_beta_tilde<-1

  #----------------------------------------------------------------------------------#


  #theta_tau2:####

  theta_tau2<-1

  #----------------------------------------------------------------------------------#

  #b (nq x 1):####

  b_list<-list()

  for(i in 1:n){

    b_list[[i]]<-matrix(0,
                        nrow=q,
                        ncol=1)

  }

  b<-dplyr::bind_rows(lapply(b_list,as.data.frame))%>%
    as.matrix

  #----------------------------------------------------------------------------------#

  #sigma2:####

  sigma2<-1

  #----------------------------------------------------------------------------------#

  #s2####

  s2<-1

  #----------------------------------------------------------------------------------#

  #Sigma (J x J):####

  Sigma<-diag(J)

  #----------------------------------------------------------------------------------#


  #tau_l2 (p x 1):####

  tau_l2<-matrix(1,
                 nrow=p,
                 ncol=1)

  #----------------------------------------------------------------------------------#

  #G (q x q)####

  G<-diag(q)

  #----------------------------------------------------------------------------------#

  #MCMC Info.:####

  # Number of MCMC Iterations

  nsim<-nsim

  # Thinning interval

  thin<-thin

  # Burnin

  burn<-burn

  #Total MCMC iterations:

  total_sim<-(nsim-burn)/thin

  #---------------------------------------------------------------------------------#


  #MCMC storage data objects:#####

  print('MCMC data storage objects')

  #For labeling MCMC storage objects:

  time_labels<-paste('t',1:J,sep='')

  #---------------------------------------------------------------------------------#

  #theta_beta_tilde:####

  theta_beta_tilde_mcmc_df<-data.frame(value=0,
                                       feature_group_size=NA,
                                       feature_group=NA,
                                       subject=NA,
                                       parameter='theta_beta_tilde',
                                       row_occasion=NA,
                                       occasion=NA,
                                       q=NA,
                                       row_q=NA,
                                       feature=NA)

  theta_beta_tilde_mcmc_df<-
    lapply(1:total_sim, function(x) theta_beta_tilde_mcmc_df)%>%
    dplyr::bind_rows()

  theta_beta_tilde_mcmc_df$mcmc_iter<-1:total_sim


  #---------------------------------------------------------------------------------#

  #theta_tau2:#####

  theta_tau2_mcmc_df<-data.frame(value=0,
                                 feature_group_size=NA,
                                 feature_group=NA,
                                 occasion=NA,
                                 subject=NA,
                                 parameter='theta_tau2',
                                 row_occasion=NA,
                                 q=NA,
                                 row_q=NA,
                                 feature=NA)

  theta_tau2_mcmc_df<-
    lapply(1:total_sim, function(x) theta_tau2_mcmc_df)%>%
    dplyr::bind_rows()

  theta_tau2_mcmc_df$mcmc_iter<-1:total_sim

  #---------------------------------------------------------------------------------#

  #sigma2:#####

  sigma2_mcmc_df<-data.frame(value=0,
                             feature_group_size=NA,
                             feature_group=NA,
                             subject=NA,
                             parameter='sigma2',
                             row_occasion=NA,
                             occasion=NA,
                             q=NA,
                             row_q=NA,
                             feature=NA)

  sigma2_mcmc_df<-
    lapply(1:total_sim, function(x) sigma2_mcmc_df)%>%
    dplyr::bind_rows()

  sigma2_mcmc_df$mcmc_iter<-1:total_sim


  #---------------------------------------------------------------------------------#

  #s2:#####

  s2_mcmc_df<-data.frame(value=0,
                         feature_group_size=NA,
                         feature_group=NA,
                         subject=NA,
                         parameter='s2',
                         row_occasion=NA,
                         occasion=NA,
                         q=NA,
                         row_q=NA,
                         feature=NA)

  s2_mcmc_df<-
    lapply(1:total_sim, function(x) s2_mcmc_df)%>%
    dplyr::bind_rows()

  s2_mcmc_df$mcmc_iter<-1:total_sim



  #---------------------------------------------------------------------------------#

  #Sigma (J X J):####

  Sigma_mcmc_df<-
    as.data.frame(matrix(rep(0,J^2),
                         nrow=J,
                         ncol=J))

  names(Sigma_mcmc_df)<-
    time_labels

  Sigma_mcmc_df$row_occasion<-
    time_labels

  Sigma_mcmc_df<-
    Sigma_mcmc_df%>%
    tidyr::gather(occasion,value,-row_occasion)%>%
    dplyr::mutate(feature_group_size=NA,
                  feature_group=NA,
                  subject=NA,
                  parameter='Sigma',
                  q=NA,
                  row_q=NA,
                  feature=NA)



  Sigma_mcmc_df<-
    lapply(1:total_sim, function(x) Sigma_mcmc_df)%>%
    dplyr::bind_rows()

  Sigma_mcmc_df$mcmc_iter<-
    rep(1:total_sim,each=J^2)

  #---------------------------------------------------------------------------------#

  #G (q X q):####

  G_mcmc_df<-
    as.data.frame(matrix(rep(0,q^2),
                         nrow=q,
                         ncol=q))

  names(G_mcmc_df)<-as.character(1:q)

  G_mcmc_df$row_q<-1:q

  G_mcmc_df<-G_mcmc_df%>%
    tidyr::gather(q,value,-row_q)%>%
    dplyr::mutate(feature_group_size=NA,
                  feature_group=NA,
                  subject=NA,
                  parameter='G',
                  feature=NA,
                  occasion=NA,
                  row_occasion=NA)


  G_mcmc_df<-
    lapply(1:total_sim, function(x) G_mcmc_df)%>%
    dplyr::bind_rows()

  G_mcmc_df$mcmc_iter<-
    rep(1:total_sim,each=q^2)

  #---------------------------------------------------------------------------------#


  #pi_0k:####

  pi_0k_mcmc_df<-data.frame(value=rep(0,K),
                            feature_group_size=p_k,
                            feature_group=1:K,
                            occasion=NA,
                            subject=NA,
                            parameter='pi_0k',
                            row_occasion=NA,
                            q=NA,
                            row_q=NA,
                            feature=NA)


  pi_0k_mcmc_df<-
    lapply(1:total_sim, function(x) pi_0k_mcmc_df)%>%
    dplyr::bind_rows()


  pi_0k_mcmc_df$mcmc_iter<-
    rep(1:total_sim,each=K)


  #---------------------------------------------------------------------------------#

  #pi_0l:####

  pi_0l_mcmc_df<-data.frame(value=rep(0,p),
                            feature_group_size=NA,
                            feature_group=NA,
                            occasion=NA,
                            subject=NA,
                            parameter='pi_0l',
                            row_occasion=NA,
                            q=NA,
                            row_q=NA,
                            feature=1:p)


  pi_0l_mcmc_df<-
    lapply(1:total_sim, function(x) pi_0l_mcmc_df)%>%
    dplyr::bind_rows()


  pi_0l_mcmc_df$mcmc_iter<-
    rep(1:total_sim,each=p)


  #---------------------------------------------------------------------------------#

  #b_i:####

  b_mcmc_df<-
    as.data.frame(matrix(0,
                         nrow=q,
                         ncol=n))

  names(b_mcmc_df)<-as.character(1:n)

  b_mcmc_df<-b_mcmc_df%>%
    dplyr::mutate(q=1:q)%>%
    tidyr::gather(subject,value,-q)%>%
    dplyr::mutate(
      feature_group_size=NA,
      feature_group=NA,
      parameter='b',
      row_occasion=NA,
      occasion=NA,
      row_q=NA,
      feature=NA)

  b_mcmc_df<-
    lapply(1:total_sim, function(x) b_mcmc_df)%>%
    dplyr::bind_rows()


  b_mcmc_df$mcmc_iter<-
    rep(1:total_sim,each=(n*q))

  #---------------------------------------------------------------------------------#

  #Beta:####

  #No need to output Beta_tilde:

  Beta_mcmc_list<-list()

  for(k in 1:K){

    k_select<-X_dim_df[k,]

    p_select<-k_select$lag_cum_p_k:k_select$cum_p_k

    Beta_mcmc_list[[k]]<-
      matrix(0,
             nrow=length(p_select),
             ncol=J)

    Beta_mcmc_list[[k]]<-
      as.data.frame(Beta_mcmc_list[[k]])

    names(Beta_mcmc_list[[k]])<-time_labels

    Beta_mcmc_list[[k]]$feature_group_size<-k_select$p_k

    Beta_mcmc_list[[k]]<-Beta_mcmc_list[[k]]%>%
      tidyr::gather(occasion,value,-feature_group_size)

    #-------------------------------#

    p_k_index<-k_select$lag_cum_p_k:k_select$cum_p_k

    p_k_index<-as.character(p_k_index)

    #-------------------------------#


    Beta_mcmc_list[[k]]$feature=as.numeric(p_k_index)

    Beta_mcmc_list[[k]]<-Beta_mcmc_list[[k]]%>%
      dplyr::mutate(
        feature_group=k,
        subject=NA,
        parameter='beta',
        # occasion=NA,
        row_occasion=NA,
        q=NA,
        row_q=NA)

  }


  #Collapse:

  beta_mcmc_df<-
    dplyr::bind_rows(Beta_mcmc_list)

  beta_mcmc_df<-
    lapply(1:total_sim, function(x) beta_mcmc_df)%>%
    dplyr::bind_rows()


  beta_mcmc_df$mcmc_iter<-
    rep(1:total_sim,each=(p*J))


  #---------------------------------------------------------------------------------#

  #tau_l2:####

  tau_l2_mcmc_df<-data.frame(value=rep(0,p),
                             feature_group_size=NA,
                             feature_group=NA,
                             occasion=NA,
                             subject=NA,
                             parameter='tau_l2',
                             row_occasion=NA,
                             q=NA,
                             row_q=NA,
                             feature=1:p)


  tau_l2_mcmc_df<-
    lapply(1:total_sim, function(x) tau_l2_mcmc_df)%>%
    dplyr::bind_rows()

  tau_l2_mcmc_df$mcmc_iter<-
    rep(1:total_sim,each=(p))


  #---------------------------------------------------------------------------------#

  #V_k (p_k x p_k):####

  V_k<-list()

  for(k in 1:K){

    k_select<-X_dim_df[k,]

    p_select<-k_select$lag_cum_p_k:k_select$cum_p_k

    V_k[[k]]<-diag(length(p_select))

  }

  #----------------------------------------------------------------------------------#


  #Make counter re: (p*J) dimension:

  p_J_counter_df<-
    data.frame(mcmc_iter=rep(1:total_sim,each=(p*J)),
               row_num=1:((p*J) * total_sim))


  #-------------------------#

  #Make counter re: (n*q) dimension:

  n_q_counter_df<-
    data.frame(mcmc_iter=rep(1:total_sim,each=(n*q)),
               row_num=1:((n*q) * total_sim))

  #-------------------------#

  #Make counter re: p dimension:

  p_counter_df<-
    data.frame(mcmc_iter=rep(1:total_sim,each=p),
               row_num=1:(p * total_sim))

  #-------------------------#

  #Make counter re: K dimension:

  K_counter_df<-
    data.frame(mcmc_iter=rep(1:total_sim,each=K),
               row_num=1:(K * total_sim))

  #-------------------------#

  #Make counter re: (q)^2 dimension:

  q_squared_counter_df<-
    data.frame(mcmc_iter=rep(1:total_sim,each=(q)^2),
               row_num=1:((q)^2 * total_sim))


  #-------------------------#

  #Make counter re: C^2 dimension:

  J_squared_counter_df<-
    data.frame(mcmc_iter=rep(1:total_sim,each=J^2),
               row_num=1:(J^2 * total_sim))


  #----------------------------------------------------------------------------------#


  #######################################################

  #Start Gibbs sampler:####

  #######################################################

  print('MCMC Sampling')

  #-------------------------#

  for (mcmc in 1:nsim){

    #Define once here re: comp. time:

    solve_Sigma<-solve(Sigma)

    #-------------------------#

    #Progress bar:

    if(mcmc %% mcmc_div == 0){
      cat(mcmc)
      cat("..")
    }

    #-------------------------#

    #Start k loop:

    for(k in 1:K){

      #Relevant indices for p_k:

      k_select<-X_dim_df[k,]

      p_select<-k_select$lag_cum_p_k:k_select$cum_p_k


      #----------------------------------------------------------------------------------#


      #Update V_k (p_k x p_k) via tau_l2:####

      diag_V_k<-tau_l2[p_select]

      if(p_k[k]>1){

        V_k[[k]]<-diag(c(diag_V_k))

      }else{

        V_k[[k]]<-diag_V_k

      }

      #----------------------------------------------------------------------------------#

      #Define I_J kron V_k:

      V_k_star<-
        (I_J%x%V_k[[k]])

      #-------------------------------#

      #Define I_pk:

      I_pk<-diag(p_k[k])

      #-------------------------------#

      #Calculate Omega_k (Cp_k x Cp_k):

      Omega_k<-
        (
          V_k_star%*%
            t(X_STAR_k[[k]])%*%
            X_STAR_k[[k]]%*%
            V_k_star
        )+
        (
          sigma2*
            (solve_Sigma%x%I_pk)
        )


      #Calculate inverse once to save comp time:

      inv_Omega_k<-solve(Omega_k)

      #-------------------------------#

      #Calculate z_k (Jn x 1):

      z_k<-
        vec_Y_transpose-
        (
          X_STAR_NOT_k[[k]]%*%beta_NOT_k[[k]]+
            Z_mat%*%b
        )


      #-------------------------------#

      #Calculate f_k (1 x 1):

      f_k<-
        t(z_k)%*%
        X_STAR_k[[k]]%*%
        V_k_star%*%
        inv_Omega_k%*%
        V_k_star%*%
        t(X_STAR_k[[k]])%*%
        z_k

      #----------------------------------------------------------------------------------#

      #Update pi_0k (K x 1):####

      #Calculate numerator of pi_0k posterior prob. parameter:

      pi_0k_prob_num<-
        theta_beta_tilde*
        (sigma2)^(0.5*J*p_k[k])*
        (
          det(Sigma)
        )^(-p_k[k]/2)*
        exp(
          (1/(2*sigma2))*f_k
        )*
        (
          det(
            inv_Omega_k
          )
        )^(1/2)


      #Convert to numeric value from dense matrix:

      pi_0k_prob_num<-as.numeric(pi_0k_prob_num)


      if(pi_0k_prob_num==Inf){

        pi_0k_prob_num<-
          10^20

      }



      pi_0k[k]<-
        rbinom(
          n=1,
          size=1,
          prob=pi_0k_prob_num/((1-theta_beta_tilde)+pi_0k_prob_num)
        )


      #----------------------------------------------------------------------------------#

      #Update beta_tilde_k (p_k*J x 1):####

      #N.b. rcpp code does not like sparse matrices, so
      #transform posterior mean and vcv matrix back
      #to dense matrices here:


      post_mean_beta_tilde_k<-
        (inv_Omega_k%*%
           V_k_star%*%
           t(X_STAR_k[[k]])%*%
           z_k
        )%>%
        as.matrix()

      post_cov_beta_tilde_k<-
        (sigma2*
           inv_Omega_k)%>%
        as.matrix()

      #-------------------------------#

      beta_tilde_k[[k]]<-

        (

          (pi_0k[k])*

            MASS::mvrnorm(
              n=1,
              mu=post_mean_beta_tilde_k,

              Sigma=post_cov_beta_tilde_k
            )%>%
            matrix(.,
                   nrow=p_k[k]*J,
                   ncol=1)

          +

            (1-pi_0k[k])*

            matrix(0,
                   nrow=(p_k[k]*J),
                   ncol=1)

        )


      #-------------------------------#

      #Calculate Beta_tilde_k (p_k x J) via updated beta_tilde_k (p_k*J x 1):

      Beta_tilde_k[[k]]<-
        beta_tilde_k[[k]]%>%
        matrix(.,
               nrow=p_k[k],
               ncol=J)

      #-------------------------------#

      #Then, calculate Beta_k (p_k x J) via V_k (p_k x p_k) :

      Beta_k[[k]]<-
        V_k[[k]]%*%
        Beta_tilde_k[[k]]


      #Calculate beta_k (p_kC x 1)

      beta_k[[k]]<-matrix(Beta_k[[k]],
                          nrow=p_k[k]*J,
                          ncol=1)

      #----------------------------------------------------------------------------------#


    }#end of k loop for Gibbs sampler


    #Start l loop for Gibbs sampler:

    for(l in 1:p){

      #Calculate z_l (Jn x 1):

      z_l<-
        vec_Y_transpose-
        (X_STAR_NOT_l[[l]]%*%
           beta_NOT_l[[l]]+
           Z_mat%*%b)

      #-------------------------#

      #Calculate m_l (nJ x1):

      m_l<-
        X_STAR_l[[l]]%*%
        beta_tilde_l[[l]]

      #Define m_l^T once here to improve comp time:

      m_lT<-t(m_l)

      #-------------------------#

      #Calculate sigma_l_inverse (1x1)

      sigma_l_inverse<-
        s2/
        (
          sigma2+
            (s2*m_lT%*%m_l)
        )

      #-------------------------#


      #Calculate f_l (1 x 1):

      f_l<-
        t(z_l)%*%
        m_l%*%
        sigma_l_inverse%*%
        m_lT%*%
        z_l

      #Turn into a numeric var. from a dense matrix:

      f_l<-as.numeric(f_l)

      #----------------------------------------------------------------------------------#

      #Update tau_l2:####

      tau_l2[l]<-

        (

          (pi_0l[l])*

            truncnorm::rtruncnorm(n=1,
                                  a=0,
                                  b=Inf,
                                  mean=
                                    sigma_l_inverse*
                                    m_lT%*%
                                    z_l,
                                  sd=
                                    sqrt(
                                      sigma2*
                                        sigma_l_inverse
                                    )
            )+

            (1-pi_0l[l])*

            matrix(0,
                   nrow=1,
                   ncol=1)

        )


      #----------------------------------------------------------------------------------#

      #Update pi_0l:####

      #Calculate numerator of pi_0l posterior prob. parameter:

      pi_0l_prob_num<-

        theta_tau2*
        2*
        sqrt(sigma2/s2)*
        exp(
          f_l/(2*sigma2)
        )*
        sqrt(
          sigma_l_inverse
        )*
        pnorm(
          q=sqrt(f_l/sigma2),
          mean=0,
          sd=1
        )


      #Turn into a numeric var. from a dense matrix:

      pi_0l_prob_num<-as.numeric(pi_0l_prob_num)

      if(pi_0l_prob_num==Inf){

        pi_0l_prob_num<-
          10^20

      }



      pi_0l[l]<-

        rbinom(
          n=1,
          size=1,
          prob=
            pi_0l_prob_num/
            ((1-theta_tau2)+pi_0l_prob_num)
        )

      #----------------------------------------------------------------------------------#


    }#end of l loop for Gibbs sampler





    #----------------------------------------------------------------------------------#

    #Create updated Beta (p x J) via Beta_k:####

    Beta<-lapply(Beta_k,as.data.frame)%>%
      dplyr::bind_rows()%>%
      as.matrix()

    #-------------------------------#

    #Create updated beta (pJ x 1) via Beta_k:

    beta<-
      matrix(Beta,
             nrow=p*J,
             ncol=1)

    #-------------------------------#

    #Update Beta_tilde (p x J) for Sigma IW update:

    Beta_tilde<-lapply(Beta_tilde_k,as.data.frame)%>%
      dplyr::bind_rows()%>%
      as.matrix()

    #-------------------------------#

    #Update beta_NOT_k (p-kJ x 1):

    for(k in 1:K){

      k_select<-X_dim_df[k,]

      p_select<-k_select$lag_cum_p_k:k_select$cum_p_k

      p_NOT_select<-which(((1:p)%in%p_select)==F)

      Beta_NOT_k<-Beta[p_NOT_select,]%>%
        matrix(.,
               nrow=length(p_NOT_select),
               ncol=J)

      beta_NOT_k[[k]]<-matrix(Beta_NOT_k,
                              nrow=length(p_NOT_select)*J,
                              ncol=1)

    }


    #-------------------------------#

    #Update beta_NOT_l ((p-1)J x 1):

    for(l in 1:p){

      p_select<-l

      p_NOT_select<-which(((1:p)%in%p_select)==F)

      Beta_NOT_l<-Beta[p_NOT_select,]%>%
        matrix(.,
               nrow=length(p_NOT_select),
               ncol=J)


      beta_NOT_l[[l]]<-matrix(Beta_NOT_l,
                              nrow=length(p_NOT_select)*J,
                              ncol=1)

    }

    #-------------------------------#

    #Update beta_tilde_l (C x 1)

    for(l in 1:p){

      p_select<-l

      beta_tilde_l[[l]]<-
        Beta_tilde[p_select,]%>%
        matrix(.,
               nrow=J,
               ncol=1)

    }


    #----------------------------------------------------------------------------------#


    #Update b (nq x 1):####

    #Define Sigma_b (nq x nq):

    Sigma_b<-
      t(Z_mat)%*%Z_mat+
      (sigma2*
         (diag(n)%x%solve(G)))

    #Define once here re: comp time:

    inv_Sigma_b<-solve(Sigma_b)

    #-------------------------------#

    #Define r_b (Jn x 1):

    r_b<-
      vec_Y_transpose-
      X_kron_mat%*%beta

    #-------------------------------#


    #N.b. rcpp code does not like sparse matrices, so
    #transform posterior mean and vcv matrix back
    #to dense matrices here:

    #Group these matrix multiplication terms
    #re: faster comp time:

    post_mean_b<-
      (inv_Sigma_b%*%
         t(Z_mat)%*%
         r_b
      )%>%
      as.matrix()

    post_cov_b<-
      (
        sigma2*
          inv_Sigma_b
      )%>%
      as.matrix()


    b<-
      MASS::mvrnorm(
        n=1,
        mu=post_mean_b,
        Sigma=post_cov_b
      )%>%
      matrix(.,
             nrow=n*q,
             ncol=1)

    #-------------------------------#

    #Create b_mat which is q x n for Gibbs update for G (q x q)

    b_mat<-matrix(b,
                  nrow=q,
                  ncol=n,
                  byrow=F)


    #----------------------------------------------------------------------------------#

    #Update G (q x q):####

    #N.b. b_mat%*%t(b_mat)
    #is equal to the sum over i of b_i%*%b_i^T

    post_scale_G<-
      C_0+
      b_mat%*%t(b_mat)

    G<-
      MCMCpack::riwish(v=
                   nu_0+n,
                 S=post_scale_G
      )


    #----------------------------------------------------------------------------------#

    #Update theta_beta_tilde (1 x 1): ####

    theta_beta_tilde<-rbeta(n=1,
                            shape1=
                              a_1+sum(pi_0k),
                            shape2=
                              b_1+K-sum(pi_0k)
    )



    #----------------------------------------------------------------------------------#


    #Update theta_tau2 (1 x 1):####

    theta_tau2<-rbeta(n=1,
                      shape1=
                        g+
                        (pi_0l%>%sum),
                      shape2=
                        h+p-
                        (pi_0l%>%sum))


    #----------------------------------------------------------------------------------#

    #Update sigma2 (1 x 1):####


    #Define r (Jn x 1)

    r<-
      vec_Y_transpose-
      (X_kron_mat%*%beta+Z_mat%*%b)

    #Use crossprod to calculate r'r here re: comp. time :

    rtr<-
      crossprod(r)%>%
      as.numeric

    #-------------------------------#

    sigma2<-MCMCpack::rinvgamma(n=1,
                                shape=
                                  alpha+(.5*n*J),
                                scale=
                                  gamma+(.5*rtr)
    )

    #----------------------------------------------------------------------------------#

    #Update Sigma (J x J):####


    #Identify for which Beta_tilde_k pi_0k = 1

    p_k_selected<-p_k[(pi_0k==1)]


    post_scale_Sigma<-
      (t(Beta_tilde)%*%Beta_tilde)+
      Q


    Sigma<-
      MCMCpack::riwish(v=
                   d+
                   sum(p_k_selected),
                 S=post_scale_Sigma
      )



    #----------------------------------------------------------------------#


    #Update s2 (1 x 1):####

    s2<-MCMCpack::rinvgamma(n=1,
                            shape=
                              o+
                              (pi_0l%>%sum/2),
                            scale=
                              u+
                              (
                                ((tau_l2)^2%>%sum)/2
                              )
    )

    #This can become Inf if sum(pi_0l)=0

    if(is.infinite(s2)==T){

      s2<-10^20

    }

    #-----------------------------------------------------------------------#

    #-----------------------------------------------------------------------#

    # Store MCMC samples: #####

    #-----------------------------------------------------------------------#

    #-----------------------------------------------------------------------#








    if (mcmc>burn && mcmc%%thin==0) {

      #-----------------------------------------------------------------------#

      #theta_beta_tilde:

      theta_beta_tilde_mcmc_df$value[(mcmc-burn)/thin]<-
        theta_beta_tilde

      #---------------------------------------------------------------------#

      #theta_tau2:

      theta_tau2_mcmc_df$value[(mcmc-burn)/thin]<-
        theta_tau2

      #-----------------------------------------------------------------------#

      #sigma2:

      sigma2_mcmc_df$value[(mcmc-burn)/thin]<-
        sigma2

      #---------------------------------------------------------------#

      #s2:

      s2_mcmc_df$value[(mcmc-burn)/thin]<-
        s2

      #-----------------------------------------------------------------#

      #Sigma:

      vec_Sigma<-
        matrix(Sigma,
               nrow=J^2,
               ncol=1)

      #Appropriate subset of the MCMC output object:

      J_squared_counter_subset<-
        J_squared_counter_df%>%
        dplyr::filter(mcmc_iter==(mcmc-burn)/thin)%>%
        dplyr::select(row_num)%>%
        as.matrix%>%
        as.vector

      Sigma_mcmc_df$value[J_squared_counter_subset]<-
        vec_Sigma

      #-----------------------------------------------------------------#

      #G:

      vec_G<-
        matrix(G,
               nrow=(q)^2,
               ncol=1)

      #Appropriate subset of the MCMC output object:

      q_squared_counter_subset<-
        q_squared_counter_df%>%
        dplyr::filter(mcmc_iter==(mcmc-burn)/thin)%>%
        dplyr::select(row_num)%>%
        as.matrix%>%
        as.vector


      G_mcmc_df$value[q_squared_counter_subset]<-
        vec_G

      #-----------------------------------------------------------------#

      #tau_l2:

      p_counter_subset<-
        p_counter_df%>%
        dplyr::filter(mcmc_iter==(mcmc-burn)/thin)%>%
        dplyr::select(row_num)%>%
        as.matrix%>%
        as.vector

      tau_l2_mcmc_df$value[p_counter_subset]<-
        tau_l2

      #----------------------------------------------------------------------------------#

      #pi_0k:

      K_counter_subset<-
        K_counter_df%>%
        dplyr::filter(mcmc_iter==(mcmc-burn)/thin)%>%
        dplyr::select(row_num)%>%
        as.matrix%>%
        as.vector


      pi_0k_mcmc_df$value[K_counter_subset]<-
        pi_0k

      #----------------------------------------------------------------------------------#

      #pi_0l:

      pi_0l_mcmc_df$value[p_counter_subset]<-
        pi_0l

      #----------------------------------------------------------------------------------#

      #b:

      vec_b<-
        matrix(b_mat,
               nrow=(n*q),
               ncol=1)

      #Appropriate subset of the MCMC output object:

      n_q_counter_subset<-
        n_q_counter_df%>%
        dplyr::filter(mcmc_iter==(mcmc-burn)/thin)%>%
        dplyr::select(row_num)%>%
        as.matrix%>%
        as.vector



      b_mcmc_df$value[n_q_counter_subset]<-
        vec_b

      #----------------------------------------------------------------------------------#

      #beta (pJ x 1):

      vec_beta<-
        lapply(beta_k,as.data.frame)%>%
        dplyr::bind_rows()%>%
        as.matrix%>%
        matrix(.,
               nrow=p*J,
               ncol=1)

      #Appropriate subset of the MCMC output object:

      p_J_counter_subset<-
        p_J_counter_df%>%
        dplyr::filter(mcmc_iter==(mcmc-burn)/thin)%>%
        dplyr::select(row_num)%>%
        as.matrix%>%
        as.vector

      beta_mcmc_df$value[p_J_counter_subset]<-
        vec_beta



    }#end of Store MCMC samples if statement


    #######################################################
    #End of MCMC iterations for loop
    #######################################################

  }

  #Aggregate mcmc results:

  MCMC_output_df<-rbind(beta_mcmc_df,

                        pi_0l_mcmc_df,
                        tau_l2_mcmc_df,

                        s2_mcmc_df,

                        b_mcmc_df,

                        pi_0k_mcmc_df,

                        G_mcmc_df,

                        Sigma_mcmc_df,

                        sigma2_mcmc_df,

                        theta_beta_tilde_mcmc_df,

                        theta_tau2_mcmc_df)

  #----------------------------------------------------------------------------------#

  MCMC_summary_df<-
    MCMC_output_df%>%
    dplyr::group_by(feature,
                    occasion,
                    feature_group,
                    subject,
                    parameter,
                    row_occasion,
                    q,
                    row_q)%>%
    dplyr::summarise(MCMC_mean=mean(value),
                     MCMC_median=median(value),
                     MCMC_sd=sd(value),
                     MCMC_LCL=quantile(value,probs = .05/2),
                     MCMC_UCL=quantile(value,probs=1-(.05/2)))%>%
    as.data.frame


  #-------------------------------#

  MCMC_summary_df$feature<-as.character(MCMC_summary_df$feature)

  MCMC_summary_df$subject<-as.numeric(MCMC_summary_df$subject)

  #-------------------------------#


  #Identify feature parameters
  #that have MCMC_median != 0

  selected_features_mcmc<-
    MCMC_summary_df%>%
    dplyr::filter(parameter=='beta',
                  MCMC_median!=0)%>%
    dplyr::select(feature)%>%
    unlist%>%
    unique%>%
    as.numeric

  #-------------------------------#


  #Print comp. time:

  print(
    (Sys.time()-start_comp_time)%>%
      as.numeric(.,units='secs')
  )


  #-------------------------------#

  #Add comp time:

  MCMC_output_df$comp_time<-
    (Sys.time()-start_comp_time)%>%
    as.numeric(.,units='secs')


  MCMC_summary_df$comp_time<-
    (Sys.time()-start_comp_time)%>%
    as.numeric(.,units='secs')


  #-------------------------------#

  #Get rid of feature_group_size col here:

  MCMC_output_df<-
    MCMC_output_df%>%
    dplyr::select(-feature_group_size)

  #Make row_q a character var.:

  MCMC_output_df<-
    MCMC_output_df%>%
    dplyr::mutate(row_q=as.character(row_q))

  MCMC_summary_df<-
    MCMC_summary_df%>%
    dplyr::mutate(row_q=as.character(row_q))

  #Make subject a character var:

  MCMC_summary_df$subject<-
    as.character(MCMC_summary_df$subject)


  #-------------------------------#

  if(raw_MCMC_output==T){

    return(

      list(

        'MCMC_output'=MCMC_output_df,

        'MCMC_summary'=MCMC_summary_df

      )
    )

  }else{#for raw_MCMC_output==F

    return(

      list(

        'MCMC_summary'=MCMC_summary_df

      )
    )

  }#end of raw_MCMC_output if/else statement


}#End of BayesianLMMFS function



