#' @title Fitting Functional Neural Networks
#'
#' @description
#' This is the main function in the `FNN` package. This function fits models of the form: f(z, b(x)) where
#' z are the scalar covariates and b(x) are the functional covariates. The form of f() is that of a neural network
#' with a generalized input space.
#'
#' @return The following are returned:
#'
#' `model` -- Full keras model that can be used with any functions that act on keras models.
#'
#' `data` -- Adjust data set after scaling and appending of scalar covariates.
#'
#' `fnc_basis_num` -- A return of the original input; describes the number of functions used in each of the k basis expansions.
#'
#' `fnc_type` -- A return of the original input; describes the basis expansion used to make the functional weights.
#'
#' `parameter_info` -- Information associated with hyperparameter choices in the model.
#'
#' `per_iter_info` -- Change in error over training iterations
#'
#' `func_obs` -- In the case when `raw_data` is True, the user may want to see the internally developed functional observations.
#' This returns those functions.
#'
#' @details Updates coming soon.
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
#' @param dropout Keras parameter that randomly drops some percentage of the neurons in a given layer.
#' If TRUE, then 0.1*layer_number will be dropped; instead, you can specify a vector equal to the number
#' of layers specifying what percentage to drop in each layer.
#'
#' @examples
#' # First, an easy example with raw_data = T
#'
#' # Loading in data
#' data("daily")
#'
#' # Functional covariates
#' temp = t(daily$tempav)
#' precip = t(daily$precav)
#' longtidunal_dat = list(temp, precip)
#'
#' # Scalar Response
#' total_prec = apply(daily$precav, 2, mean)
#'
#' # Running model
#' fit1 = fnn.fit(resp = total_prec,
#'                func_cov = longtidunal_dat,
#'                scalar_cov = NULL,
#'                learn_rate = 0.0001,
#'                raw_data = TRUE)
#'
#' # Classification Example with raw_data = TRUE
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
#' # Running prediction, gets probabilities
#' predict_class = fnn.predict(fit_class,
#'                             func_cov = func_covs_test,
#'                             scalar_cov = scalar_test,
#'                             domain_range = list(c(850, 1050)),
#'                             raw_data = TRUE)
#'
#' # Example with Pre-Processing (raw_data = F)
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
#' # Prediction example can be seen with ?fnn.fit()
#'
#' # Functional Response Example:
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
#' # Getting data into proper format
#' ind = 1:30
#' nbasis = 65
#' weather_data_train <- array(dim = c(nbasis, ncol(temp_data), 1))
#' weather_data_train[,,1] = temp_data
#' scalar_train = data.frame(weather_scalar[,1])
#' resp_train = t(resp_mat)
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
#'
#'
#' @export
#' @import keras tensorflow fda.usc fda ggplot2 ggpubr pbapply reshape2 flux Matrix doParallel
#' @importFrom caret createFolds
#' @importFrom caret confusionMatrix
#' @importFrom stats predict

#returns product of two numbers, as a trivial example
fnn.fit <- function(resp,
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
                    raw_data = FALSE,
                    dropout = FALSE){

  # Checking what kind of problem it is
  if(is.factor(resp) == TRUE){
    resp2 = as.numeric(as.character(resp))
    resp = to_categorical(resp)
    problem_type = "classification"
  } else {
    resp2 = resp
    problem_type = "regression"
  }

  #### Output size
  if(is.vector(resp) == TRUE){
    output_size = 1
  } else {
    output_size = ncol(resp)
  }

  #### FLAG: The above might break for functional responses

  #### Error Checks

  if(length(domain_range) != length(num_basis)){
    stop("The number of domain ranges doesn't match length of num_basis")
  }

  if(length(domain_range) != length(basis_choice)){
    stop("The number of domain ranges doesn't match number of basis choices")
  }

  if(length(num_basis) != length(basis_choice)){
    stop("Too many/few num_basis - doesn't match number of basis choices")
  }

  if(hidden_layers != length(neurons_per_layer)){
    stop("The number of hidden layers doesn't match the dimension of neuron vector")
  }

  if(hidden_layers != length(activations_in_layers)){
    stop("The number of hidden layers doesn't match the dimension of activation vector")
  }

  if(length(activations_in_layers) != length(neurons_per_layer)){
    stop("Too many/few activations/neurons")
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

    for (t in 1:dim_check) {

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

  ##### Helper Functions #####

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

  # Looping to get approximations
  print(paste0("Evaluating Integrals:"))
  for (i in 1:dim(func_cov)[3]) {

    # Current data set
    df <- func_cov[,,i]

    # Turning into matrix
    if(is.vector(df) == TRUE){
      print('yes')
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

  # Now we have the data set to pass onto the network, we can set up the data so that it is well suited to be
  # passed onto the network. This means normalizing things and rewriting some other things

  # Normalize training data
  if(covariate_scaling == TRUE){
    train_x <- scale(converted_df)
  } else {
    train_x <- as.matrix(cbind(converted_df[,c(1:sum(num_basis))], scale(converted_df[,-c(1:sum(num_basis))])))
  }

  train_y <- resp

  # Now, we can move onto creating the model. This means taking advantage of the last three variables. We will use another
  # function to do this that lets us add layers easily.


  if(is.vector(resp2) == TRUE & dropout == FALSE){

    # Creating model
    build_model <- function(train_x,
                            neurons_per_layer,
                            activations_in_layers,
                            hidden_layers,
                            output_size,
                            loss_choice,
                            metric_choice) {

      # Initializing model for FNN layer
      model <- keras_model_sequential() %>%
        layer_dense(units = neurons_per_layer[1], activation = activations_in_layers[1],
                    input_shape = dim(train_x)[2])

      # Adding in additional model layers
      if(hidden_layers > 1){
        for (i in 1:(hidden_layers - 1)) {
          model <- model %>% layer_dense(units = neurons_per_layer[i + 1], activation = activations_in_layers[i + 1])
        }
      }

      # Setting up final layer
      if(problem_type != "classification"){
        model <- model %>% layer_dense(units = output_size)
      } else {
        model <- model %>% layer_dense(units = ncol(resp),
                                       activation = 'softmax')
      }


      # Setting up other model parameters
      model %>% compile(
        loss = loss_choice,
        optimizer = optimizer_adam(lr = learn_rate, decay = decay_rate),
        metrics = metric_choice
      )

      return(model)
    }

    # Now we have the model set up, we can begin to initialize the network before it is ultimately trained. This will also
    # print out a summary of the model thus far
    model <- build_model(train_x,
                         neurons_per_layer,
                         activations_in_layers,
                         hidden_layers,
                         output_size,
                         loss_choice,
                         metric_choice)

    if(print_info ==  T){
      print(model)
    }

    # We can also display the progress of the network to make it easier to visualize using the following. This is
    # borrowed from the keras write up for R on the official website
    print_dot_callback <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        if (epoch %% 80 == 0) cat("\n")
        cat("x")
      }
    )

    # The patience parameter is the amount of epochs to check for improvement.
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience_param)

    # Now finally, we can fit the model
    if(early_stopping == TRUE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop, print_dot_callback)
      )
    } else if(early_stopping == TRUE & print_info == FALSE) {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop)
      )
    } else if(early_stopping == FALSE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(print_dot_callback)
      )
    } else {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list()
      )
    }


    # Plotting the errors
    if(print_info == TRUE){
      print(plot(history, metrics = "mean_squared_error", smooth = FALSE) +
              theme_bw() +
              xlab("Epoch Number") +
              ylab(""))
    }

    # Skipping line
    cat("\n")

    # Printing out
    if(print_info == TRUE){
      print(history)
    }
  }

  if(is.vector(resp2) == FALSE & func_resp_method == 1 & dropout == FALSE){

    # Creating model
    build_model <- function(train_x,
                            neurons_per_layer,
                            activations_in_layers,
                            hidden_layers,
                            output_size,
                            loss_choice,
                            metric_choice) {

      # Initializing model for FNN layer
      model <- keras_model_sequential() %>%
        layer_dense(units = neurons_per_layer[1], activation = activations_in_layers[1],
                    input_shape = dim(train_x)[2])

      # Adding in additional model layers
      if(hidden_layers > 1){
        for (i in 1:(hidden_layers - 1)) {
          model <- model %>% layer_dense(units = neurons_per_layer[i + 1], activation = activations_in_layers[i + 1])
        }
      }

      # Setting up final layer
      model <- model %>% layer_dense(units = output_size)

      # Setting up other model parameters
      model %>% compile(
        loss = loss_choice,
        optimizer = optimizer_adam(lr = learn_rate, decay = decay_rate),
        metrics = metric_choice
      )

      return(model)
    }

    # Now we have the model set up, we can begin to initialize the network before it is ultimately trained. This will also
    # print out a summary of the model thus far
    model <- build_model(train_x,
                         neurons_per_layer,
                         activations_in_layers,
                         hidden_layers,
                         output_size,
                         loss_choice,
                         metric_choice)

    if(print_info ==  TRUE){
      print(model)
    }

    # We can also display the progress of the network to make it easier to visualize using the following. This is
    # borrowed from the keras write up for R on the official website
    print_dot_callback <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        if (epoch %% 80 == 0) cat("\n")
        cat("x")
      }
    )

    # The patience parameter is the amount of epochs to check for improvement.
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience_param)

    # Now finally, we can fit the model
    if(early_stopping == TRUE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop, print_dot_callback)
      )
    } else if(early_stopping == TRUE & print_info == FALSE) {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop)
      )
    } else if(early_stopping == FALSE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(print_dot_callback)
      )
    } else {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list()
      )
    }


    # Plotting the errors
    if(print_info == TRUE){
      print(plot(history, metrics = "mean_squared_error", smooth = FALSE) +
              theme_bw() +
              xlab("Epoch Number") +
              ylab(""))
    }

    # Skipping line
    cat("\n")

    # Printing out
    if(print_info == TRUE){
      print(history)
    }



  } else {

  }

  if(is.vector(resp2) == TRUE & dropout != FALSE){

    # Creating model
    build_model <- function(train_x,
                            neurons_per_layer,
                            activations_in_layers,
                            hidden_layers,
                            output_size,
                            loss_choice,
                            metric_choice) {

      # Initializing model for FNN layer
      model <- keras_model_sequential() %>%
        layer_dense(units = neurons_per_layer[1], activation = activations_in_layers[1],
                    input_shape = dim(train_x)[2])

      # Adding in additional model layers
      if(dropout == TRUE){
        if(hidden_layers > 1){
          for (i in 1:(hidden_layers - 1)) {
            model <- model %>% layer_dropout(rate = (hidden_layers - i) * 0.1) %>%
              layer_dense(units = neurons_per_layer[i + 1], activation = activations_in_layers[i + 1])
          }
        }
      } else {
        if(hidden_layers > 1){
          for (i in 1:(hidden_layers - 1)) {
            model <- model %>% layer_dropout(rate = dropout[i]) %>%
              layer_dense(units = neurons_per_layer[i + 1], activation = activations_in_layers[i + 1])
          }
        }
      }


      # Setting up final layer
      if(problem_type != "classification"){
        model <- model %>% layer_dense(units = output_size)
      } else {
        model <- model %>% layer_dense(units = ncol(resp),
                                       activation = 'softmax')
      }


      # Setting up other model parameters
      model %>% compile(
        loss = loss_choice,
        optimizer = optimizer_adam(lr = learn_rate, decay = decay_rate),
        metrics = metric_choice
      )

      return(model)
    }

    # Now we have the model set up, we can begin to initialize the network before it is ultimately trained. This will also
    # print out a summary of the model thus far
    model <- build_model(train_x,
                         neurons_per_layer,
                         activations_in_layers,
                         hidden_layers,
                         output_size,
                         loss_choice,
                         metric_choice)

    if(print_info ==  T){
      print(model)
    }

    # We can also display the progress of the network to make it easier to visualize using the following. This is
    # borrowed from the keras write up for R on the official website
    print_dot_callback <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        if (epoch %% 80 == 0) cat("\n")
        cat("x")
      }
    )

    # The patience parameter is the amount of epochs to check for improvement.
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience_param)

    # Now finally, we can fit the model
    if(early_stopping == TRUE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop, print_dot_callback)
      )
    } else if(early_stopping == TRUE & print_info == FALSE) {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop)
      )
    } else if(early_stopping == FALSE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(print_dot_callback)
      )
    } else {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list()
      )
    }


    # Plotting the errors
    if(print_info == TRUE){
      print(plot(history, metrics = "mean_squared_error", smooth = FALSE) +
              theme_bw() +
              xlab("Epoch Number") +
              ylab(""))
    }

    # Skipping line
    cat("\n")

    # Printing out
    if(print_info == TRUE){
      print(history)
    }
  }

  if(is.vector(resp2) == FALSE & func_resp_method == 1 & dropout == TRUE){

    # Creating model
    build_model <- function(train_x,
                            neurons_per_layer,
                            activations_in_layers,
                            hidden_layers,
                            output_size,
                            loss_choice,
                            metric_choice) {

      # Initializing model for FNN layer
      model <- keras_model_sequential() %>%
        layer_dense(units = neurons_per_layer[1], activation = activations_in_layers[1],
                    input_shape = dim(train_x)[2])

      # Adding in additional model layers
      if(dropout == TRUE){
        if(hidden_layers > 1){
          for (i in 1:(hidden_layers - 1)) {
            model <- model %>% layer_dropout(rate = (hidden_layers - i) * 0.1) %>%
              layer_dense(units = neurons_per_layer[i + 1], activation = activations_in_layers[i + 1])
          }
        }
      } else {
        if(hidden_layers > 1){
          for (i in 1:(hidden_layers - 1)) {
            model <- model %>% layer_dropout(rate = dropout[i]) %>%
              layer_dense(units = neurons_per_layer[i + 1], activation = activations_in_layers[i + 1])
          }
        }
      }


      # Setting up final layer
      model <- model %>% layer_dense(units = output_size)

      # Setting up other model parameters
      model %>% compile(
        loss = loss_choice,
        optimizer = optimizer_adam(lr = learn_rate, decay = decay_rate),
        metrics = metric_choice
      )

      return(model)
    }

    # Now we have the model set up, we can begin to initialize the network before it is ultimately trained. This will also
    # print out a summary of the model thus far
    model <- build_model(train_x,
                         neurons_per_layer,
                         activations_in_layers,
                         hidden_layers,
                         output_size,
                         loss_choice,
                         metric_choice)

    if(print_info ==  TRUE){
      print(model)
    }

    # We can also display the progress of the network to make it easier to visualize using the following. This is
    # borrowed from the keras write up for R on the official website
    print_dot_callback <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        if (epoch %% 80 == 0) cat("\n")
        cat("x")
      }
    )

    # The patience parameter is the amount of epochs to check for improvement.
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience_param)

    # Now finally, we can fit the model
    if(early_stopping == TRUE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop, print_dot_callback)
      )
    } else if(early_stopping == TRUE & print_info == FALSE) {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(early_stop)
      )
    } else if(early_stopping == FALSE & print_info == TRUE){
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list(print_dot_callback)
      )
    } else {
      history <- model %>% fit(
        train_x,
        train_y,
        epochs = epochs,
        validation_split = val_split,
        verbose = 0,
        callbacks = list()
      )
    }


    # Plotting the errors
    if(print_info == TRUE){
      print(plot(history, metrics = "mean_squared_error", smooth = FALSE) +
              theme_bw() +
              xlab("Epoch Number") +
              ylab(""))
    }

    # Skipping line
    cat("\n")

    # Printing out
    if(print_info == TRUE){
      print(history)
    }



  }

  # Returning the model
  return(list(model = model,
              data = train_x,
              fnc_basis_num = num_basis,
              fnc_type = basis_choice,
              parameter_info = history$params,
              per_iter_info = history$metrics,
              func_obs = func_cov))
}
