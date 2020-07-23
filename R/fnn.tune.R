#' @title Tuning Functional Neural Networks
#'
#' @description
#' A convenience function for the user that implements a simple grid search for the purpose of tuning. For each combination
#' in the grid, a cross-validated error is calculated. The best combination is returned along with additional information.
#' This function only works for scalar responses.
#'
#' @return The following are returned:
#'
#' `Parameters` -- The final list of hyperparameter chosen by the tuning process.
#'
#' `All_Information` -- A list object containing the errors for every combination in the grid. Each element of the list
#' corresponds to a different choice of number of hidden layers.
#'
#' `Best_Per_Layer` -- An object that returns the best parameter combination for each choice of hidden layers.
#'
#' `Grid_List` -- An object containing information about all combinations tried by the tuning process.
#'
#' @details No additional details for now.
#'
#' @param tune_list This is a list object containing the values from which to develop the grid. For each of the hyperparameters
#' that can be tuned for (`num_hidden_layers`, `neurons`, `epochs`, `val_split`, `patience`, `learn_rate`, `num_basis`,
#' `activation_choice`), the user inputs a set of values to try. Note that the combinations are found based on the number of
#' hidden layers. For example, if `num_hidden_layers` = 3 and `neurons` = c(8, 16), then the combinations will begin as
#' c(8, 8, 8), c(8, 8, 16), ..., c(16, 16, 16). Example provided below.
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
#' @param domain_range List of size k. Each element of the list is a 2-dimensional vector containing the upper and lower
#' bounds of the k-th functional weight.
#'
#' @param batch_size Size of the batch for stochastic gradient descent.
#'
#' @param decay_rate A modification to the learning rate that decreases the learning rate as more and more learning
#' iterations are completed.
#'
#' @param nfolds The number of folds to be used in the cross-validation process.
#'
#' @param cores For the purpose of parallelization.
#'
#' @param raw_data If True, then user does not need to create functional observations beforehand. The function will
#' internally take care of that pre-processing.
#'
#' @examples
#' # libraries
#' library(fda)
#'
#' # Loading data
#' data("daily")
#'
#' # Obtaining response
#' total_prec = apply(daily$precav, 2, mean)
#'
#' # Creating functional data
#' temp_data = array(dim = c(65, 35, 1))
#' tempbasis65  = create.fourier.basis(c(0,365), 65)
#' timepts = seq(1, 365, 1)
#' temp_fd = Data2fd(timepts, daily$tempav, tempbasis65)
#'
#' # Data set up
#' temp_data[,,1] = temp_fd$coefs
#'
#' # Creating grid
#' tune_list_weather = list(num_hidden_layers = c(2),
#'                          neurons = c(8, 16),
#'                          epochs = c(250),
#'                          val_split = c(0.2),
#'                          patience = c(15),
#'                          learn_rate = c(0.01, 0.1),
#'                          num_basis = c(7),
#'                          activation_choice = c("relu", "sigmoid"))
#'
#' # Running Tuning
#' weather_tuned = fnn.tune(tune_list_weather,
#'                          total_prec,
#'                          temp_data,
#'                          basis_choice = c("fourier"),
#'                          domain_range = list(c(1, 24)),
#'                          nfolds = 2)
#'
#' # Looking at results
#' weather_tuned

#'
#' @export
# @import keras tensorflow fda.usc fda ggplot2 ggpubr caret pbapply reshape2 flux Matrix doParallel

#returns product of two numbers, as a trivial example
fnn.tune = function(tune_list,
                    resp,
                    func_cov,
                    scalar_cov = NULL,
                    basis_choice,
                    domain_range,
                    batch_size = 32,
                    decay_rate = 0,
                    nfolds = 5,
                    cores = 4,
                    raw_data = FALSE){

  # Parallel apply set up
  #plan(multiprocess, workers = cores)

  #### Output size
  if(is.vector(resp) == TRUE){
    output_size = 1
  } else {
    output_size = ncol(resp)
  }

  if(raw_data == TRUE){
    dim_check = length(func_cov)
  } else {
    dim_check = dim(func_cov)[3]
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
      curr_domain = domain_range[[1]] # BE CAREFUL HERE - ALL DOMAINS NEED TO BE THE SAME IN THIS CASE

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


  if(output_size == 1){

    # Setting up function
    tune_func = function(x,
                         nfolds,
                         resp,
                         func_cov,
                         scalar_cov,
                         basis_choice,
                         domain_range,
                         batch_size,
                         decay_rate,
                         raw_data){

      # Setting seed
      use_session_with_seed(
        1,
        disable_gpu = FALSE,
        disable_parallel_cpu = FALSE,
        quiet = TRUE
      )

      # Clearing irrelevant information
      colnames(x) <- NULL
      rownames(x) <- NULL

      # Running model
      model_results = fnn.cv(nfolds,
                             resp,
                             func_cov = func_cov,
                             scalar_cov = scalar_cov,
                             basis_choice = basis_choice,
                             num_basis = as.numeric(as.character((x[(current_layer + 1):(length(basis_choice) + current_layer)]))),
                             hidden_layers = current_layer,
                             neurons_per_layer = as.numeric(as.character(x[(length(basis_choice) + current_layer + 1):((length(basis_choice) + current_layer) + current_layer)])),
                             activations_in_layers = as.character(x[1:current_layer]),
                             domain_range = domain_range,
                             epochs = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 1])),
                             loss_choice = "mse",
                             metric_choice = list("mean_squared_error"),
                             val_split = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 2])),
                             learn_rate = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 4])),
                             patience_param = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 3])),
                             early_stopping = TRUE,
                             print_info = FALSE,
                             batch_size = batch_size,
                             decay_rate = decay_rate,
                             raw_data = FALSE)

      # Putting together
      list_returned <- list(MSPE = model_results$MSPE$Overall_MSPE,
                            num_basis = as.numeric(as.character((x[(current_layer + 1):(length(basis_choice) + current_layer)]))),
                            hidden_layers = current_layer,
                            neurons_per_layer = as.numeric(as.character(x[(length(basis_choice) + current_layer + 1):((length(basis_choice) + current_layer) + current_layer)])),
                            activations_in_layers = as.character(x[1:current_layer]),
                            epochs = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 1])),
                            val_split = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 2])),
                            patience_param = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 3])),
                            learn_rate = as.numeric(as.character(x[((length(basis_choice) + current_layer) + current_layer) + 4])))

      # Clearing backend
      K <- backend()
      K$clear_session()

      # Returning
      return(list_returned)

    }

    # Saving MSPEs
    Errors = list()
    All_Errors = list()
    Grid_List = list()

    # Setting up tuning parameters
    for (i in 1:length(tune_list$num_hidden_layers)) {

      # Current layer number
      current_layer = tune_list$num_hidden_layers[i]

      # Creating data frame of list
      df = expand.grid(rep(list(tune_list$neurons), tune_list$num_hidden_layers[i]), stringsAsFactors = FALSE)
      df2 = expand.grid(rep(list(tune_list$num_basis), length(basis_choice)), stringsAsFactors = FALSE)
      df3 = expand.grid(rep(list(tune_list$activation_choice), tune_list$num_hidden_layers[i]), stringsAsFactors = FALSE)
      colnames(df2)[length(basis_choice)] <- "Var2.y"
      colnames(df3)[i] <- "Var2.z"

      # Getting grid
      pre_grid = expand.grid(df$Var1,
                             Var2.y = df2$Var2.y,
                             Var2.z = df3$Var2.z,
                             tune_list$epochs,
                             tune_list$val_split,
                             tune_list$patience,
                             tune_list$learn_rate)

      # Merging
      combined <- unique(merge(df, pre_grid, by = "Var1"))
      combined2 <- unique(merge(df2, combined, by = "Var2.y"))
      final_grid <- suppressWarnings(unique(merge(df3, combined2, by = "Var2.z")))

      # Saving grid
      Grid_List[[i]] = final_grid

      # Now, we can pass on the combinations to the model
      results = pbapply(final_grid, 1, tune_func,
                        nfolds = nfolds,
                        resp = resp,
                        func_cov = func_cov,
                        scalar_cov = scalar_cov,
                        basis_choice = basis_choice,
                        domain_range = domain_range,
                        batch_size = batch_size,
                        decay_rate = decay_rate,
                        raw_data = FALSE)

      # Initializing
      MSPE_vals = c()

      # Collecting results
      for (u in 1:length(results)) {
        MSPE_vals[u] <- as.vector(results[[u]][1])
      }

      # All Errors
      All_Errors[[i]] = results

      # Getting best
      Errors[[i]] = results[[which.min(do.call(c, MSPE_vals))]]

      # Printing where we are at
      cat("\n")
      print(paste0("Done tuning for: ", current_layer, " hidden layers."))

    }

    # Initializing
    MSPE_after = c()

    # Getting best set of parameters
    for (i in 1:length(tune_list$num_hidden_layers)) {
      MSPE_after[i] = Errors[[i]]$MSPE
    }

    # Selecting minimum
    best = which.min(MSPE_after)

    # Returning best set of parameters
    return(list(Parameters = Errors[[best]],
                All_Information = All_Errors,
                Best_Per_Layer = Errors,
                Grid_List = Grid_List))

  } else {

    print("Tuning isn't available yet for functional responses")

    return()

  }

}
