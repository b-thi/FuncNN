#' @title Functional Neural Networks with Cross-validation
#'
#' @description
#' This is a convenience function for the user. The inputs are largely the same as the [fnn.fit()] function with the
#' additional parameter of fold choice. This function only works for scalar responses.
#'
#' @return The following are returned.
#'
#' `predicted_folds` -- The predicted scalar values in each fold.
#'
#' `true_folds` -- The true values of the response in each fold.
#'
#' `MSPE` -- A list object containing the MSPE in each fold and the overall cross-validated MSPE.
#'
#' `fold_indices` -- The generated indices for each fold; for replication purposes.
#'
#' @details No additional details for now.
#'
#' @param nfolds The number of folds to be used in the cross-validation process.
#'
#' @param resp For scalar responses, this is a vector of the observed dependent variable. For functional responses,
#' this is a matrix where each row contains the basis coefficients defining the functional response (for each observation).
#'
#' @param func_cov The form of this depends on whether the `raw_data` argument is true or not. If true, then this is
#' a list of k matrices. The dimensionality of the matrices should be the same (n x p) where n is the number of
#' observations and p is the number of longitudinal observations. If `raw_data` is false, then the input should be a tensor
#' with dimensionality b x n x k where b is the number of basis functions used to define the functional covariates, n is
#' the number of observations, and k is the number of functional covariates.
#'
#' @param scalar_cov A matrix contained the multivariate information associated with the data set. This is all of your
#' non-longitudinal data.
#'
#' @param basis_choice A vector of size k (the number of functional covariates) with either "fourier" or "bspline" as the inputs.
#' This is the choice for the basis functions used for the functional weight expansion. If you only specify one, with k > 1,
#' then the argument will repeat that choice for all k functional covariates.
#'
#' @param num_basis A vector of size k defining the number of basis functions to be used in the basis expansion. Must be odd
#' for `fourier` basis choices. If you only specify one, with k > 1, then the argument will repeat that choice for all
#' k functional covariates.
#'
#' @param hidden_layers The number of hidden layers to be used in the neural network.
#'
#' @param neurons_per_layer Vector of size = `hidden_layers`. The u-th element of the vector corresponds to the number of neurons
#' in the u-th hidden layer.
#'
#' @param activations_in_layers Vector of size = `hidden_layers`. The u-th element of the vector corresponds to the
#' activation choice in the u-th hidden layer.
#'
#' @param domain_range List of size k. Each element of the list is a 2-dimensional vector containing the upper and lower
#' bounds of the k-th functional weight.
#'
#' @param epochs The number of training iterations.
#'
#' @param loss_choice This parameter defines the loss function used in the learning process.
#'
#' @param metric_choice This parameter defines the printed out error metric.
#'
#' @param val_split A parameter that decides the percentage split of the inputted data set.
#'
#' @param learn_rate Hyperparameter that defines how quickly you move in the direction of the gradient.
#'
#' @param patience_param A keras parameter that decides how many additional `epochs` are eclipsed with minimal change in
#' error before the learning process is stopped. This is only active if `early_stopping = TRUE`
#'
#' @param early_stopping If True, then learning process will be halted early if error improvement isn't seen.
#'
#' @param print_info If True, function will output information about the model as it is trained.
#'
#' @param batch_size Size of the batch for stochastic gradient descent.
#'
#' @param decay_rate A modification to the learning rate that decreases the learning rate as more and more learning
#' iterations are completed.
#'
#' @param func_resp_method Set to 1 by default. In the future, this will be set to 2 for an alternative functional response
#' approach.
#'
#' @param covariate_scaling If True, then data will be internally scaled before model development.
#'
#' @param raw_data If True, then user does not need to create functional observations beforehand. The function will
#' internally take care of that pre-processing.
#'
#' @examples
#' # Libraries
#' library(fda)
#'
#' # Loading data
#' data("daily")
#'
#' # Creating functional data
#' nbasis = 65
#' temp_data = array(dim = c(nbasis, 35, 1))
#' tempbasis65  = create.fourier.basis(c(0,365), nbasis)
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
#' # Setting up data to pass in to function
#' weather_data_full <- array(dim = c(nbasis, ncol(temp_data), 1))
#' weather_data_full[,,1] = temp_data
#' scalar_full = data.frame(weather_scalar[,1])
#' total_prec = apply(daily$precav, 2, mean)
#'
#' # cross-validating
#' cv_example <- fnn.cv(nfolds = 5,
#'                      resp = total_prec,
#'                      func_cov = weather_data_full,
#'                      scalar_cov = scalar_full,
#'                      domain_range = list(c(1, 365)),
#'                      learn_rate = 0.001)
#'
#' @export
# @import keras tensorflow fda.usc fda ggplot2 ggpubr pbapply reshape2 flux Matrix doParallel

#returns product of two numbers, as a trivial example
fnn.cv <- function(nfolds,
                   resp,
                   func_cov,
                   scalar_cov = NULL,
                   basis_choice = c("fourier"),
                   num_basis = c(7),
                   hidden_layers = 2,
                   neurons_per_layer = c(64, 64),
                   activations_in_layers = c("sigmoid", "linear"),
                   domain_range = list(c(0, 1)),
                   epochs = 100,
                   loss_choice = "mse",
                   metric_choice = list("mean_squared_error"),
                   val_split = 0.2,
                   learn_rate = 0.001,
                   patience_param = 15,
                   early_stopping = TRUE,
                   print_info = T,
                   batch_size = 32,
                   decay_rate = 0,
                   func_resp_method = 1,
                   covariate_scaling = TRUE,
                   raw_data = FALSE){

  #### Output size
  if(is.vector(resp) == TRUE){
    output_size = 1
  } else {
    output_size = ncol(resp)
  }

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
    num_basis = rep(num_basis, length(func_cov))

    # Fixing basis type
    basis_choice = rep(basis_choice, length(func_cov))

    # Final update to domain range
    domain_range = domain_range_list

    # Warning
    # print("Warning: You only specified basis information for one functional covariate -- it will be repeated for all functional covariates")

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

  # Creating folds
  folds <- createFolds(resp, k = nfolds, list = TRUE, returnTrain = FALSE)

  # Predictions initialization
  predictions = list()
  true_values = list()
  MSPE_Fold = list()
  differences = c()

  # Looping to create models
  for (i in 1:nfolds) {

    if(output_size == 1){

      if(dim(func_cov)[3] == 1){

        # Initializing arrays
        data_train <- array(data = NA, dim = c(dim(func_cov)[1],
                                               do.call(sum, lapply(folds, length)) - length(folds[[i]]),
                                               dim(func_cov)[3]))

        data_test <- array(data = NA, dim = c(dim(func_cov)[1],
                                              length(folds[[i]]),
                                              dim(func_cov)[3]))

        # Splitting into test and train
        data_train[,,1] = func_cov[, -folds[[i]], ]
        data_test[,,1] = func_cov[, folds[[i]], ]
        resp_train = resp[-folds[[i]]]
        resp_test = resp[folds[[i]]]

      } else if (dim(func_cov)[2] == nfolds) {

        # Initializing
        data_test = array(data = NA, dim = c(dim(func_cov)[1],
                                             1,
                                             dim(func_cov)[3]))

        # Splitting into test and train
        data_train = func_cov[, -folds[[i]], ]
        data_test[,1,] = func_cov[, folds[[i]], ]
        resp_train = resp[-folds[[i]]]
        resp_test = resp[folds[[i]]]
      } else {

        # Splitting into test and train
        data_train = func_cov[, -folds[[i]], ]
        data_test = func_cov[, folds[[i]], ]
        resp_train = resp[-folds[[i]]]
        resp_test = resp[folds[[i]]]
      }

      # Saving
      true_values[[i]] = resp_test

    } else {

      stop("Cross validation for functional responses is still under development")


    }

    # Running model
    model_cv = fnn.fit(resp = resp_train,
                   func_cov = data_train,
                   scalar_cov = scalar_cov,
                   basis_choice = basis_choice,
                   num_basis = num_basis,
                   hidden_layers = hidden_layers,
                   neurons_per_layer = neurons_per_layer,
                   activations_in_layers = activations_in_layers,
                   domain_range = domain_range,
                   epochs = epochs,
                   loss_choice = loss_choice,
                   metric_choice = metric_choice,
                   val_split = val_split,
                   learn_rate = learn_rate,
                   patience_param = patience_param,
                   early_stopping = early_stopping,
                   print_info = print_info,
                   batch_size = batch_size,
                   decay_rate = decay_rate,
                   func_resp_method = func_resp_method,
                   covariate_scaling= covariate_scaling,
                   raw_data = FALSE,
                   dropout = FALSE)

    # Predicting
    predictions[[i]] = fnn.predict(model = model_cv,
                                   func_cov = data_test,
                                   scalar_cov = scalar_cov,
                                   basis_choice = basis_choice,
                                   num_basis = num_basis,
                                   domain_range = domain_range,
                                   covariate_scaling = covariate_scaling,
                                   raw_data = FALSE)

    # Getting differences
    differences = c(differences, predictions[[i]] - resp_test)

    # Getting MSPE per fold
    MSPE_Fold[[i]] = mean((predictions[[i]] - true_values[[i]])^2)

    # Folds done
    if(print_info == TRUE){
      cat("\n")
      print(paste0("Folds Done: ", i))
    }

  }

  # Calculating MSPE
  MSPE_to_return = mean(differences^2)

  # Clearing backend
  K <- backend()
  K$clear_session()


  # Returning
  return(list(predicted_folds = predictions,
              true_folds = true_values,
              MSPE = list(Overall_MSPE = MSPE_to_return,
                          Fold_MSPEs = MSPE_Fold),
              fold_indices = folds))

}
