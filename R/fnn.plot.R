#' @title Plotting Functional Response Predictions
#'
#' @description
#' This function is to be used for functional responses. It outputs a `ggplot()` object of the predicted functional responses.
#'
#' @return The following are returned:
#'
#' `plot` -- A `ggplot()` object of the predicted functional responses.
#'
#' `evaluations` -- The discrete evaluations across the domain of the functional response.
#'
#' @details No additional details for now.
#'
#' @param FNN_Predict_Object An object output by the `fnn.predict()` function. Must be for when the problem is that of
#' a functional response.
#'
#' @param Basis_Type The type of basis to use to create the functional response.
#'
#' @param domain_range The continuum range of the functional responses.
#'
#' @param step_size The size of the movement from the lower bound of the `domanin_range` to the upper bound.
#'
#' @examples
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
#' # Getting predictions
#' predictions = fnn.predict(weather_func_fnn,
#'                           weather_data_test,
#'                           scalar_cov = scalar_test,
#'                           basis_choice = c("bspline"),
#'                           num_basis = c(7),
#'                           domain_range = list(c(1, 365)))
#'
#' # Looking at plot
#' fnn.plot(predictions, domain_range = c(1, 365), step_size = 1, Basis_Type = "bspline")
#'
#' @export
# @import keras tensorflow fda.usc fda ggplot2 ggpubr caret pbapply reshape2 flux Matrix doParallel

#returns product of two numbers, as a trivial example
fnn.plot <- function(FNN_Predict_Object,
                     Basis_Type = "fourier",
                     domain_range = c(0, 1),
                     step_size = 0.01){

  # This function will print out curves

  if(Basis_Type == "fourier"){

    # Creating basis
    tempbasis = create.fourier.basis(domain_range, ncol(FNN_Predict_Object))

    # Creating null data fd
    test_fd = suppressWarnings(Data2fd(seq(domain_range[1], domain_range[2], step_size),
                                       y = seq(domain_range[1], domain_range[2], step_size),
                                       tempbasis))

    # Adding in coefficients
    test_fd$coefs = t(FNN_Predict_Object)

    # Getting evaluations
    fd_evals = eval.fd(evalarg = seq(domain_range[1], domain_range[2], step_size), fdobj = test_fd)
    fd_ds = melt(fd_evals)
    row.names(fd_ds) = NULL
    colnames(fd_ds) = c("Continuum", "Obs", "Value")

    # Plotting functional observations
    print(ggplot(data = fd_ds, aes(x = fd_ds$Continuum, y = fd_ds$Value, color = as.factor(fd_ds$Obs))) +
            geom_line(size = 1.25) +
            theme_bw() +
            labs(color='Obs Number') +
            labs(x = "s", y = "x(s)", title = "Functional Curves") +
            theme(plot.title = element_text(hjust = 0.5)))

    final_plot = ggplot(data = fd_ds, aes(x = fd_ds$Continuum, y = fd_ds$Value, color = as.factor(fd_ds$Obs))) +
      geom_line(size = 1.25) +
      theme_bw() +
      labs(color='Obs Number') +
      labs(x = "s", y = "x(s)", title = "Functional Curves") +
      theme(plot.title = element_text(hjust = 0.5))

    evaluations = fd_evals

  }

  if(Basis_Type == "bspline"){

    if(ncol(FNN_Predict_Object) > 3){
      order_chosen_beta = 4
    } else {
      order_chosen_beta = 1
    }

    # Creating basis
    tempbasis = create.bspline.basis(domain_range, ncol(FNN_Predict_Object),
                                     norder = order_chosen_beta)

    # Creating null data fd
    test_fd = suppressWarnings(Data2fd(seq(domain_range[1], domain_range[2], step_size),
                                       y = seq(domain_range[1], domain_range[2], step_size),
                                       tempbasis))

    # Adding in coefficients
    test_fd$coefs = t(FNN_Predict_Object)

    # Getting evaluations
    fd_evals = eval.fd(evalarg = seq(domain_range[1], domain_range[2], step_size), fdobj = test_fd)
    fd_ds = melt(fd_evals)
    row.names(fd_ds) = NULL
    colnames(fd_ds) = c("Continuum", "Obs", "Value")

    # Plotting functional observations
    print(ggplot(data = fd_ds, aes(x = fd_ds$Continuum, y = fd_ds$Value, color = as.factor(fd_ds$Obs))) +
            geom_line(size = 1.25) +
            theme_bw() +
            labs(color='Obs Number') +
            labs(x = "s", y = "x(s)", title = "Functional Curves") +
            theme(plot.title = element_text(hjust = 0.5)))

    final_plot = ggplot(data = fd_ds, aes(x = fd_ds$Continuum, y = fd_ds$Value, color = as.factor(fd_ds$Obs))) +
      geom_line(size = 1.25) +
      theme_bw() +
      labs(color='Obs Number') +
      labs(x = "s", y = "x(s)", title = "Functional Curves") +
      theme(plot.title = element_text(hjust = 0.5))

    evaluations = fd_evals

  }

  return(list(plot = final_plot, values = evaluations))

}
