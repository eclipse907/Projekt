from onevariable_function import OnevariableFunction
import main
import optimisation as opt


"""
    TO EXECUTE THIS PROGRAM, YOU NEED TO HAVE INSTALLED THESE LIBRARIES:

    https://numpy.org/
    https://www.sympy.org/

    BOTH ARE NEEDED FOR PLOTTING AND CALCULATING PURPOSES
"""

"""
This module's role is to present proper usage of OnevariableFunction class and valid input of 
arguments in command line.


CHOOSING OPTIMISING ALGORITHM, FUNCTION TO OPTIMISE AND PARAMETERS

Optimising algorithm and the function are the only parameters needed for this program to work. Other parameters will
be set to their default values if not added.
    Example:
        python main.py -algo SGD -ip x^2-2*x+1
    
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


Function MUST be passed in the previously mentioned format or any other similar format that can be recognized as Python functions.
    Examples of valid formats:
        x^2-2*x+1
        x^2 - 2*x + 1
        x**2 - 2*x + 1
        5*sin(x)


Function can also be passed as a parameter to functions in optimizator.py and optimisation.py modules.

Plotting is optional, it can be disabled by setting is_to_be_plotted value to False.

"""

# Uncomment this section for a preview of the optimisation using OnevariableFunction class:


value = "x*sin(1/x)"

opt_function = OnevariableFunction(value)

algorithm = "SGDM"

learning_rate = 0.4
initial_point = -2
num_of_iterations = 200
is_to_be_plotted = True
momentum = 0.1

opt.optimise(algorithm, opt_function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum, beta1=0.9,
             beta2=0.999)


