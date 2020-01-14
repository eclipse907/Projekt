from polynomial_function import PolynomialFunction
import main
import optimisation as opt

"""
This module's role is to present proper usage of PolynomialFunction class and valid input of 
arguments in command line.


CHOOSING OPTIMISING ALGORITHM, FUNCTION TO OPTIMISE AND PARAMETERS

Optimising algorithm and the function are the only parameters needed for this program to work. Other parameters will
be set to their default values.
    Example:
        python main.py -algo SGD -ip x^2-2x+1
    
    This example will execute the optimisation of the said polynomial function using SGD algorithm.
    
Parameters need to be added using special commands:
    -algo           Choosing an optimising algorithm *
    -if             Choosing a function to optimise *
    -lr             Choosing learning rate for the algorithm
    -ip             Choosing the initial point for the optimisation
    -it             Choosing the number of iterations
    -mom            Choosing the momentum -
    -b1             Choosing exponential decay for the 1st moment +
    -b2             Choosing exponential decay for the 2nd moment +
    
    *  - These parameters are needed for the program to work
    -  - These parameters are used only for SGDM optimation
    +  - These parameters are used only for ADAM optimisation


Function MUST be passed in the previously mentioned format, but spaces are also allowed.#
    Examples of valid formats:
        x^2-2x+1
        x^2 - 2x + 1
        x ^ 2 - 2 x + 1

"""

# Uncomment this for a preview of the optimisation using PolynomialFunction class:

"""
algorithm = "SGD"
function = PolynomialFunction("x^2-2x+1")
learning_rate = 0.01
initial_point = 10
num_of_iterations = 200
is_to_be_plotted = True
momentum = 0.9


opt.optimise(algorithm, function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum, beta1=0, beta2=0)

"""
