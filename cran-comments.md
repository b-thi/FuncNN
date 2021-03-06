## FuncNN Package Information
This is the second submission of this package to CRAN.  For this reason, we have tentatively included instructions on how to install the package from CRAN in our README, which have been commented out.

## Test environments
* local Windows 10, R 3.6.3
* local Windows 7, R 3.6.3
* local OS X, R 3.6.3

## R CMD check results
There were no ERRORs, WARNINGs or NOTEs. 

## Addressing concerns of first submission:

### - All T and F have been changed to TRUE and FALSE
### - We have rewritten our citation to deal with formatting
### - While we have rewritten some print statements as messages, we could not rewrite all of them in this was as some are based on dependencies. As a workaround to easily silence a function, we provide an option for the functions in the form of "print_info". Setting this to FALSE will silence the functions.
### - Our examples don't run in under 5 seconds (the slowest one being about 6 seconds) due to some of the pre-processing required. Also, our package requires TensorFlow to be installed on the machine.

## Addressing concerns of second submission:

### - We have checked again and hopefully all T and F have been changed this time to TRUE and FALSE
### - We have discussed with Gregor Seyer why we can't unwrap the examples. 