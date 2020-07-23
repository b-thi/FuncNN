#' @title Output of Estimated Functional Weights
#'
#' @description
#' This function outputs plots and `ggplot()` objects of the functional weights found by the `fnn.fit()` model.
#'
#' @return The following are returned:
#'
#' `FNC_Coefficients` -- The estimated coefficients defining the basis expansion for each of the k functional weights.
#'
#' `saved_plot` -- A list of size k of `ggplot()` objects.
#'
#' @details No additional details for now.
#'
#' @param model A keras model as outputted by `fnn.fit()`.
#'
#' @param domain_range List of size k. Each element of the list is a 2-dimensional vector containing the upper and lower
#' bounds of the k-th functional weight. Must be the same covariates as input into `fnn.fit()`.
#'
#' @param covariate_scaling If True, then data will be internally scaled before model development.
#'
#' @examples
#' # libraries
#' library(fda)
#'
#' # loading data
#' tecator = FNN::tecator
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
#' # Getting data ready to pass into function
#' ind = 1:165
#' tec_data_train <- array(dim = c(nbasis, length(ind), 3))
#' tec_data_train = tecator_data[, ind, ]
#' tecResp_train = tecator_resp[ind]
#' scalar_train = data.frame(tecator_scalar[ind,1])
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
#' # Functional weights for this model
#' est_func_weights = fnn.fnc(tecator_fnn, domain_range = list(c(850, 1050),
#'                                                             c(850, 1050),
#'                                                             c(850, 1050)))
#'
#' @export
# @import keras tensorflow fda.usc fda ggplot2 ggpubr caret pbapply reshape2 flux Matrix doParallel


#returns product of two numbers, as a trivial example
fnn.fnc = function(model, domain_range, covariate_scaling = FALSE){

  # Getting weights
  weights = rowMeans(get_weights(model$model)[[1]])

  # Creating list to separate
  fnc_list = list()

  # Separating out for multiple FNCs
  for (i in 1:length(model$fnc_basis_num)) {

    # Current number of basis and choice of basis information
    cur_basis_num <- model$fnc_basis_num[i]

    # Storing previous numbers
    if(i == 1){
      left_end = 1
      right_end = cur_basis_num
    } else {
      left_end = sum(model$fnc_basis_num[1:(i - 1)]) + 1
      right_end = (left_end - 1) + cur_basis_num
    }

    fnc_list[[i]] = weights[left_end:right_end]
  }

  # Function to make fnc values
  fnc_valuations = function(obs_weight, cur_basis, cur_domain){

    # Doing for fourier basis
    if(cur_basis == "fourier"){

      # Creating basis
      fnc_basis = create.fourier.basis(rangeval = c(cur_domain[1], cur_domain[2]),
                                       nbasis = length(obs_weight))

      # Getting evaluations
      fnc_evals = eval.basis(evalarg = seq(cur_domain[1], cur_domain[2], length.out = 500),
                             basisobj = fnc_basis)

      # Getting valuations
      output_vals = c(obs_weight%*%t(fnc_evals))

      # Returning
      return(output_vals)

    }

    # Doing for bspline basis
    if(cur_basis == "bspline"){

      if(length(obs_weight) > 3){
        order_chosen = 4
      } else {
        order_chosen = length(obs_weight)
      }

      # Creating basis
      fnc_basis = create.bspline.basis(rangeval = c(cur_domain[1], cur_domain[2]),
                                       nbasis = length(obs_weight),
                                       norder = order_chosen)

      # Getting evaluations
      fnc_evals = eval.basis(evalarg = seq(cur_domain[1], cur_domain[2], length.out = 500),
                             basisobj = fnc_basis)


      # Getting valuations
      output_vals = c(obs_weight%*%t(fnc_evals))

      # Returning
      return(output_vals)

    }

  }

  # Initializing plot list
  plots_saved = list()

  # Looping to get FNCs
  for (j in 1:length(fnc_list)) {

    # Current basis type
    current_basis = model$fnc_type[j]

    # Current range
    current_range = domain_range[[j]]

    # Getting values
    if(covariate_scaling == FALSE){

      vals = fnc_valuations(obs_weight = fnc_list[[j]],
                            cur_basis = current_basis,
                            cur_domain = current_range)

      # Getting updated function
      beta_coef_fnn <- data.frame(continuum = seq(current_range[1], current_range[2], length.out = 500),
                                  beta_evals = vals)

    } else {

      dom_scale = c(scale(current_range))

      vals = fnc_valuations(obs_weight = fnc_list[[j]],
                            cur_basis = current_basis,
                            cur_domain = dom_scale)

      # Getting updated function
      beta_coef_fnn <- data.frame(continuum = seq(dom_scale[1], dom_scale[2], length.out = 500),
                                  beta_evals = vals)

    }

    # ggplot return
    plots_saved[[j]] = beta_coef_fnn %>%
      ggplot(aes(x = beta_coef_fnn$continuum, y = beta_coef_fnn$beta_evals)) +
      geom_line(size = 1.5, color = "blue") +
      theme_bw() +
      xlab("Continuum") +
      ylab("beta(s)") +
      ggtitle(paste0("Functional Neural Coefficient: ", j))
    theme(plot.title = element_text(hjust = 0.5)) +
      theme(axis.text=element_text(size=12),
            axis.title=element_text(size=14,face="bold"))
  }

  # Printing Plots
  print(ggarrange(plotlist = plots_saved, ncol = length(fnc_list), nrow = 1))

  # Returning information
  return(list(FNC_Coefficients = fnc_list,
              saved_plot = plots_saved))
}

