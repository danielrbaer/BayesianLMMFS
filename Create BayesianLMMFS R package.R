#Packages:

library("devtools")
library('roxygen2')
library("mathjaxr")
library('dplyr')
library('ggplot2')
library("knitr")
library('Rdpack')

#--------------------------------------------------------------------#

# Create the framework for the R package:####

here::dr_here()

devtools::create("BayesianLMMFS")

#--------------------------------------------------------------------#

# Add function to R package which fits our model:

# All R functions in this R package belong in the R directory.

#--------------------------------------------------------------------#

#Include Rcpp code into R package:

compileAttributes("mypack")


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
