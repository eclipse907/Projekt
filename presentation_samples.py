from onevariable_function import OnevariableFunction
import optimisation as opt

# Stochastic gradient descent (SGD)

"""

#Uncomment this section for a preview

value = "x*sin(1/x)"

opt_function = OnevariableFunction(value)

algorithm = "SGD"

learning_rate = 0.2
initial_point = -3
num_of_iterations = 200
is_to_be_plotted = True

opt.optimise(algorithm, opt_function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum=0, beta1=0.9,
             beta2=0.999)

"""


# Stochastic gradient descent with momentum (SGDM)

"""
#Uncomment this section for a preview

value = "x*sin(1/x)"

opt_function = OnevariableFunction(value)

algorithm = "SGDM"

learning_rate = 0.3
initial_point = -3
num_of_iterations = 200
is_to_be_plotted = True
momentum = 0.01

opt.optimise(algorithm, opt_function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum, beta1=0.9,
             beta2=0.999)

"""

# ADAM

"""

#Uncomment this section for a preview

value = "x*sin(1/x)"

opt_function = OnevariableFunction(value)

algorithm = "ADAM"

learning_rate = 0.1
initial_point = -3
num_of_iterations = 200
is_to_be_plotted = True

opt.optimise(algorithm, opt_function, learning_rate, initial_point, num_of_iterations, is_to_be_plotted, momentum=0, beta1=0.9,
             beta2=0.999)
"""

