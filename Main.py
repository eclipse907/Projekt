#!/usr/bin/env python3

from PolynomialFunction import PolynomialFunction
import Optimisation as opt
import argparse


# default values of parameters are to be decided, these are just for debugging purposes

def check_arguments(options):
    if options.algorithm == "SGD":
        if options.learning_rate is None:
            options.learning_rate = 0.02
        if options.initial_point is None:
            options.initial_point = 10
        if options.iterations is None:
            options.iterations = 200
    elif options.algorithm == "SGDM":
        if options.learning_rate is None:
            options.learning_rate = 0.02
        if options.initial_point is None:
            options.initial_point = 10
        if options.iterations is None:
            options.iterations = 200
    elif options.algorithm == "ADAM":
        if options.learning_rate is None:
            options.learning_rate = 0.02
        if options.initial_point is None:
            options.initial_point = 10
        if options.iterations is None:
            options.iterations = 200
    else:
        raise Exception('Illegal algorithm - choose a valid algorithm [SGD, SGDM, ADAM]')


def main():
    parser = argparse.ArgumentParser(
        description='Choose an optimising algorithm, function to optimise and parameters: learning rate, initial point and number of iterations: ')
    parser.add_argument('-algo', '--optimising algorithm', type=str, action="store", nargs='?', help="Chosen algorithm",
                        dest="algorithm")
    parser.add_argument('-if', '--input_function', type=str, action="store", nargs='?', help="Function to optimise",
                        dest="input_function")
    parser.add_argument('-lr', '--learning_rate', type=float, action="store", nargs='?',
                        help="Learning rate for the algorithm", dest="learning_rate")
    parser.add_argument('-ip', "--initial_point", type=int, action="store", nargs='?',
                        help="Initial point of the algorithm", dest="initial_point")
    parser.add_argument('-it', '--iterations', type=int, action="store", nargs='?',
                        help="Number of iterations for the algorithm", dest="iterations")
    options = parser.parse_args()

    check_arguments(options)

    function = PolynomialFunction(options.input_function)
    algorithm = options.algorithm
    learning_rate = options.learning_rate
    initial_point = options.initial_point
    num_of_iterations = options.iterations

    print("Given function: " + str(
        options.input_function) + " Choosen algorithm: " + algorithm + " Learning rate: " + str(
        learning_rate) + " Inital point: " + str(initial_point) + " Number of iterations: " + str(num_of_iterations))
    print(opt.optimise(algorithm, function, learning_rate, initial_point, num_of_iterations))


"""default values

   function = PolynomialFunction("a^3+a+1")
   opt.optimise("SGD", function, 0.1, 100, 200)
   opt.optimise("SGDM", function, 0.005, 1, 1, 0.05)
"""

if __name__ == '__main__':
    main()
