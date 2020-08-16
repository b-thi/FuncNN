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
### - We have included an example that should run in less than 5 seconds for our main fnn.fit() function.