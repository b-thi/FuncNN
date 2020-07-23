#' @title Prediction using Functional Neural Networks
#'
#' @description
#' The prediction function associated with the fnn model allowing for users to quickly get scalar or functional outputs.
#'
#' @return The following is returned:
#'
#' `Predictions` -- A vector of scalar predictions or a matrix of basis coefficients for functional responses.
#'
#' @details No additional details for now.
#'
#' @param model A keras model as outputted by `fnn.fit()`.
#'
#' @param func_cov The form of this depends on whether the `raw_data` argument is true or not. If true, then this is
#' a list of k matrices. The dimensionality of the matrices should be the same (n x p) where n is the number of
#' observations and p is the number of longitudinal observations. If `raw_data` is false, then the input should be a tensor
#' with dimensionality b x n x k where b is the number of basis functions used to define the functional covariates, n is
#' the number of observations, and k is the number of functional covariates. Must be the same covariates as input into
#' `fnn.fit()` although here, they will likely be the 'test' observations.
#'
#' @param scalar_cov A matrix contained the multivariate information associated with the data set. This is all of your
#' non-longitudinal data. Must be the same covariates as input into
#' `fnn.fit()` although here, they will likely be the 'test' observations.
#'
#' @param basis_choice A vector of size k (the number of functional covariates) with either "fourier" or "bspline" as the inputs.
#' This is the choice for the basis functions used for the functional weight expansion. If you only specify one, with k > 1,
#' then the argument will repeat that choice for all k functional covariates. Should be the same choices as input into
#' `fnn.fit()`.
#'
#' @param num_basis A vector of size k defining the number of basis functions to be used in the basis expansion. Must be odd
#' for `fourier` basis choices. If you only specify one, with k > 1, then the argument will repeat that choice for all
#' k functional covariates. Should be the same values as input into
#' `fnn.fit()`.
#'
#' @param domain_range List of size k. Each element of the list is a 2-dimensional vector containing the upper and lower
#' bounds of the k-th functional weight. Must be the same covariates as input into `fnn.fit()`.
#'
#' @param covariate_scaling If True, then data will be internally scaled before model development.
#'
#' @param raw_data If True, then user does not need to create functional observations beforehand. The function will
#' internally take care of that pre-processing.
#'
#' @examples
#' # First, we do an example with a scalar response:
#'
#' # loading data
#' tecator = FNN::tecator
#'
#' # libraries
#' library(fda)
#'
#' # define the time points on which the functional predictor is observed.
#' timepts = tecator$absorp.fdata$argvals
#'
#' # define the fourier basis
#' nbasis = 29
#' spline_basis = create.fourier.basis(tecator$absorp.fdata$rangeval, nbasis)
#'
#' # convert the functional predictor into a fda object and getting deriv
#' tecator_fd =  Data2fd(timepts, t(tecator$absorp.fdata$data), spline_basis)
#' tecator_deriv = deriv.fd(tecator_fd)
#' tecator_deriv2 = deriv.fd(tecator_deriv)
#'
#' # Non functional covariate
#' tecator_scalar = data.frame(water = tecator$y$Water)
#'
#' # Response
#' tecator_resp = tecator$y$Fat
#'
#' # Getting data into right format
#' tecator_data = array(dim = c(nbasis, length(tecator_resp), 3))
#' tecator_data[,,1] = tecator_fd$coefs
#' tecator_data[,,2] = tecator_deriv$coefs
#' tecator_data[,,3] = tecator_deriv2$coefs
#'
#' # Splitting into test and train for third FNN
#' ind = 1:165
#' tec_data_train <- array(dim = c(nbasis, length(ind), 3))
#' tec_data_test <- array(dim = c(nbasis, nrow(tecator$absorp.fdata$data) - length(ind), 3))
#' tec_data_train = tecator_data[, ind, ]
#' tec_data_test = tecator_data[, -ind, ]
#' tecResp_train = tecator_resp[ind]
#' tecResp_test = tecator_resp[-ind]
#' scalar_train = data.frame(tecator_scalar[ind,1])
#' scalar_test = data.frame(tecator_scalar[-ind,1])
#'
#' # Setting up network
#' tecator_fnn = fnn.fit(resp = tecResp_train,
#'                       func_cov = tec_data_train,
#'                       scalar_cov = scalar_train,
#'                       basis_choice = c("fourier", "fourier", "fourier"),
#'                       num_basis = c(5, 5, 7),
#'                       hidden_layers = 4,
#'                       neurons_per_layer = c(64, 64, 64, 64),
#'                       activations_in_layers = c("relu", "relu", "relu", "linear"),
#'                       domain_range = list(c(850, 1050), c(850, 1050), c(850, 1050)),
#'                       epochs = 300,
#'                       learn_rate = 0.002)
#'
#' # Predicting
#' pred_tec = fnn.predict(tecator_fnn,
#'                        tec_data_test,
#'                        scalar_cov = scalar_test,
#'                        basis_choice = c("fourier", "fourier", "fourier"),
#'                        num_basis = c(5, 5, 7),
#'                        domain_range = list(c(850, 1050), c(850, 1050), c(850, 1050)))
#'
#' # Now an example with functional responses
#'
#' # libraries
#' library(fda)
#'
#' # Loading data
#' data("daily")
#'
#' # Creating functional data
#' temp_data = array(dim = c(65, 35, 1))
#' tempbasis65  = create.fourier.basis(c(0,365), 65)
#' tempbasis7 = create.bspline.basis(c(0,365), 7, norder = 4)
#' timepts = seq(1, 365, 1)
#' temp_fd = Data2fd(timepts, daily$tempav, tempbasis65)
#' prec_fd = Data2fd(timepts, daily$precav, tempbasis7)
#' prec_fd$coefs = scale(prec_fd$coefs)
#'
#' # Data set up
#' temp_data[,,1] = temp_fd$coefs
#' resp_mat = prec_fd$coefs
#'
#' # Non functional covariate
#' weather_scalar = data.frame(total_prec = apply(daily$precav, 2, sum))
#'
#' # Splitting into test and train
#' ind = 1:30
#' nbasis = 65
#' weather_data_train <- array(dim = c(nbasis, length(ind), 1))
#' weather_data_test <- array(dim = c(nbasis, ncol(daily$tempav) - length(ind), 1))
#' weather_data_train[,,1] = temp_data[, ind, ]
#' weather_data_test[,,1] = temp_data[, -ind, ]
#' scalar_train = data.frame(weather_scalar[ind,1])
#' scalar_test = data.frame(weather_scalar[-ind,1])
#' resp_train = t(resp_mat[,ind])
#' resp_test = t(resp_mat[,-ind])
#'
#' # Running model
#' weather_func_fnn <- fnn.fit(resp = resp_train,
#'                             func_cov = weather_data_train,
#'                             scalar_cov = scalar_train,
#'                             basis_choice = c("bspline"),
#'                             num_basis = c(7),
#'                             hidden_layers = 2,
#'                             neurons_per_layer = c(1024, 1024),
#'                             activations_in_layers = c("sigmoid", "linear"),
#'                             domain_range = list(c(1, 365)),
#'                             epochs = 300,
#'                             learn_rate = 0.01,
#'                             func_resp_method = 1)
#'
#' # Getting Predictions
#' predictions = fnn.predict(weather_func_fnn,
#'                           weather_data_test,
#'                           scalar_cov = scalar_test,
#'                           basis_choice = c("bspline"),
#'                           num_basis = c(7),
#'                           domain_range = list(c(1, 365)))
#'
#' # Looking at predictions
#' predictions
#'
#' # Classification Prediction
#'
#' # Loading data
#' tecator = FNN::tecator
#'
#' # Making classification bins
#' tecator_resp = as.factor(ifelse(tecator$y$Fat > 25, 1, 0))
#'
#' # Non functional covariate
#' tecator_scalar = data.frame(water = tecator$y$Water)
#'
#' # Splitting data
#' ind = sample(1:length(tecator_resp), round(0.75*length(tecator_resp)))
#' train_y = tecator_resp[ind]
#' test_y = tecator_resp[-ind]
#' train_x = tecator$absorp.fdata$data[ind,]
#' test_x = tecator$absorp.fdata$data[-ind,]
#' scalar_train = data.frame(tecator_scalar[ind,1])
#' scalar_test = data.frame(tecator_scalar[-ind,1])
#'
#' # Making list element to pass in
#' func_covs_train = list(train_x)
#' func_covs_test = list(test_x)
#'
#' # Now running model
#' fit_class = fnn.fit(resp = train_y,
#'                     func_cov = func_covs_train,
#'                     scalar_cov = scalar_train,
#'                     hidden_layers = 6,
#'                     neurons_per_layer = c(24, 24, 24, 24, 24, 58),
#'                     activations_in_layers = c("relu", "relu", "relu", "relu", "relu", "linear"),
#'                     domain_range = list(c(850, 1050)),
#'                     learn_rate = 0.001,
#'                     epochs = 100,
#'                     raw_data = TRUE,
#'                     early_stopping = TRUE)
#'
#' # Running prediction
#' predict_class = fnn.predict(fit_class,
#'                             func_cov = func_covs_test,
#'                             scalar_cov = scalar_test,
#'                             domain_range = list(c(850, 1050)),
#'                             raw_data = TRUE)
#'
#' # Rounding predictions (they are probabilities)
#' rounded_preds = ifelse(round(predict_class)[,2] == 1, 1, 0)
#'
#' # Confusion matrix
#' # caret::confusionMatrix(as.factor(rounded_preds), as.factor(test_y))
#'
#'
#' @export
# @import keras tensorflow fda.usc fda ggplot2 ggpubr caret pbapply reshape2 flux Matrix doParallel

#returns product of two numbers, as a trivial example
fnn.predict = function(model,
                       func_cov,
                       scalar_cov = NULL,
                       basis_choice = c("fourier"),
                       num_basis = c(7),
                       domain_range = list(c(0, 1)),
                       covariate_scaling = TRUE,
                       raw_data = FALSE){

  ##### Helper Functions #####

  # Getting check for raw vs. non raw
  if(raw_data == TRUE){
    dim_check = length(func_cov)
  } else {
    dim_check = dim(func_cov)[3]
  }

  if(dim_check > length(num_basis)){

    # Fixing domain range
    domain_range_list = list()

    for (t in 1:length(func_cov)) {

      domain_range_list[[t]] = domain_range[[1]]

    }

    # Fixing num basis
    num_basis = rep(num_basis, dim_check)

    # Fixing basis type
    basis_choice = rep(basis_choice, dim_check)

    # Final update to domain range
    domain_range = domain_range_list

    # Warning
    print("Warning: You only specified basis information for one functional covariate -- it will be repeated for all functional covariates")

  }

  #### Creating functional observations in the case of raw data
  if(raw_data == TRUE){

    # Taking in data
    dat = func_cov

    # Setting up array
    temp_tensor = array(dim = c(31, nrow(dat[[1]]), length(dat)))

    for (t in 1:length(dat)) {

      # Getting appropriate obs
      curr_func = dat[[t]]

      # Getting current domain
      curr_domain = domain_range[[t]]

      # Creating basis (using bspline)
      basis_setup = create.bspline.basis(rangeval = c(curr_domain[1], curr_domain[2]),
                                         nbasis = 31,
                                         norder = 4)

      # Time points
      time_points = seq(curr_domain[1], curr_domain[2], length.out = ncol(curr_func))

      # Making functional observation
      temp_fd = Data2fd(time_points, t(curr_func), basis_setup)

      # Storing data
      temp_tensor[,,t] = temp_fd$coefs

    }

    # Saving as appropriate names
    func_cov = temp_tensor

  }

  # Integration Approximation for fourier and b-spline
  integral_eval <- function(functional_data,
                            beta_basis,
                            num_fd_basis = dim(func_cov)[1],
                            num_beta_basis,
                            range){

    if(beta_basis == "fourier"){

      # So, first the basis is created for functional, x(s)
      fourier_basis_feature = create.fourier.basis(rangeval = c(range[1], range[2]),
                                                   nbasis = num_fd_basis)

      # evaluating fourier basis for x(s)
      integ_values_fourier = eval.basis(evalarg = seq(range[1], range[2], length.out = 500),
                                        basisobj = fourier_basis_feature)

      # functional observation
      x_temp_fourier = functional_data%*%t(integ_values_fourier)

      # Now making b spline for beta(s)
      fourier_basis_beta = create.fourier.basis(rangeval = c(range[1], range[2]),
                                                nbasis = num_beta_basis)

      # evaluating functions for beta(s)
      fourier_evals = eval.basis(evalarg = seq(range[1], range[2], length.out = 500),
                                 basisobj = fourier_basis_beta)

      # Getting x(s)*beta basis function (integrand)
      eval_func_fourier = apply(fourier_evals, 2, function(x){return((x*x_temp_fourier))})

      # Getting integral
      integral_fourier = apply(eval_func_fourier, 2, function(x){
        return(auc(x = seq(range[1], range[2], length.out = 500),
                   y = x))})

      # returning
      return(integral_fourier)

    }

    if(beta_basis == "bspline"){

      if(num_fd_basis > 3){
        order_chosen_fd = 4
      } else {
        order_chosen_fd = num_fd_basis
      }

      if(num_beta_basis > 3){
        order_chosen_beta = 4
      } else {
        order_chosen_beta = num_beta_basis
      }

      # So, first the basis is created for functional, x(s)
      bspline_basis_feature = create.bspline.basis(rangeval = c(range[1], range[2]),
                                                   nbasis = num_fd_basis,
                                                   norder = order_chosen_fd)

      # evaluating b-spline basis for x(s)
      integ_values_bspline = eval.basis(evalarg = seq(range[1], range[2], length.out = 500),
                                        basisobj = bspline_basis_feature)

      # functional observation
      x_temp_bspline = functional_data%*%t(integ_values_bspline)

      # Now making b spline for beta(s)
      bspline_basis_beta = create.bspline.basis(rangeval = c(range[1], range[2]),
                                                nbasis = num_beta_basis,
                                                norder = order_chosen_beta)

      # evaluating functions for beta(s)
      bspline_evals = eval.basis(evalarg = seq(range[1], range[2], length.out = 500),
                                 basisobj = bspline_basis_beta)

      # Getting x(s)*beta basis function (integrand)
      eval_func_bspline = apply(bspline_evals, 2, function(x){return((x*x_temp_bspline))})

      # Getting integral
      integral_bspline = apply(eval_func_bspline, 2, function(x){
        return(auc(x = seq(range[1], range[2], length.out = 500),
                   y = x))})

      return(integral_bspline)
    }


  }


  # First, we need to create the proper data set. This means to get the approximations and append
  # them together for each of the covariates. We are asking for the user to pass an array where the
  # third dimension is equal to K = the number of functional covariates. Each of these will contain
  # the coefficients as found by turning the data into a functional data object.

  # Initializing matrix to keep everything inside across all functional covariates

  if(is.null(scalar_cov)){
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis)))
  } else {
    converted_df <- data.frame(matrix(nrow = dim(func_cov)[2],
                                      ncol = sum(num_basis) + ncol(scalar_cov)))
  }

  # # Looping to get approximations
  # parallel_holder <- foreach(i=1:dim(func_cov)[3], .combine=cbind) %dopar% {
  #
  #   # Current data set
  #   df <- func_cov[,,i]
  #
  #   # Turning into matrix
  #   if(is.vector(df) == T){
  #     print('yes')
  #     test_mat = matrix(nrow = length(df), ncol = 1)
  #     test_mat[,1] = df
  #     df = test_mat
  #   }
  #
  #   # Current number of basis and choice of basis information
  #   cur_basis_num <- num_basis[i]
  #   cur_basis <- basis_choice[i]
  #
  #   # Getting current range
  #   cur_range <- domain_range[[i]]
  #
  #   # Storing previous numbers
  #   if(i == 1){
  #     left_end = 1
  #     right_end = cur_basis_num
  #   } else {
  #     left_end = sum(num_basis[1:(i - 1)]) + 1
  #     right_end = (left_end - 1) + cur_basis_num
  #   }
  #
  #   converted_df[, left_end:right_end] = apply(df, 2, integral_eval, beta_basis = cur_basis,
  #                                              num_beta_basis = cur_basis_num,
  #                                              range = cur_range)
  #   converted_df[, left_end:right_end]
  # }
  #
  # # Now attaching scalar covariates
  # converted_df[, 1:sum(num_basis)] <- parallel_holder
  #
  # if(is.null(scalar_cov)){
  #   converted_df <- converted_df
  # } else{
  #   #for (k in 1:nrow(converted_df)) {
  #   #  converted_df[k, (sum(num_basis) + 1):(sum(num_basis) + ncol(scalar_cov))] <- scalar_cov[k,]
  #   #}
  #   converted_df[, (sum(num_basis) + 1):(sum(num_basis) + ncol(scalar_cov))] <- scalar_cov
  # }


  # Now we have the data set to pass onto the network, we can set up the data so that it is well suited to be
  # passed onto the network. This means normalizing things and rewriting some other things

  # Looping to get approximations
  print(paste0("Evaluating Integrals:"))
  for (i in 1:dim(func_cov)[3]) {

    # Current data set
    df <- func_cov[,,i]

    # Turning into matrix
    if(is.vector(df) == TRUE){
      test_mat = matrix(nrow = length(df), ncol = 1)
      test_mat[,1] = df
      df = test_mat
    }

    # Current number of basis and choice of basis information
    cur_basis_num <- num_basis[i]
    cur_basis <- basis_choice[i]

    # Getting current range
    cur_range <- domain_range[[i]]

    # Storing previous numbers
    if(i == 1){
      left_end = 1
      right_end = cur_basis_num
    } else {
      left_end = sum(num_basis[1:(i - 1)]) + 1
      right_end = (left_end - 1) + cur_basis_num
    }

    # Getting evaluations
    converted_df[, left_end:right_end] = t(pbapply(df, 2, integral_eval, beta_basis = cur_basis,
                                               num_beta_basis = cur_basis_num,
                                               range = cur_range))
  }

  # Now attaching scalar covariates
  if(is.null(scalar_cov)){
    converted_df <- converted_df
  } else{
    for (k in 1:nrow(converted_df)) {
      converted_df[k, (sum(num_basis) + 1):(sum(num_basis) + ncol(scalar_cov))] <- scalar_cov[k,]
    }
  }

  # Use means and standard deviations from training set to normalize test set
  if(covariate_scaling == TRUE){
    col_means_train <- attr(model$data, "scaled:center")
    col_stddevs_train <- attr(model$data, "scaled:scale")
    test_x <- scale(converted_df, center = col_means_train, scale = col_stddevs_train)
  } else {
    test_x <- converted_df
  }


  # Predicting
  test_predictions <- model$model %>% predict(test_x)

  # Returning prediction
  return(prediction = test_predictions)

}
