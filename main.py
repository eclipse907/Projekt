#!/usr/bin/env python3

from polynomial_function import PolynomialFunction
import optimisation as opt
import argparse


def check_arguments(options):
    if str(options.algorithm).upper() == "SGD":
        if options.learning_rate is None:
            options.learning_rate = 0.01
        if options.initial_point is None:
            options.initial_point = 50
        if options.iterations is None:
            options.iterations = 200
    elif str(options.algorithm).upper() == "SGDM":
        if options.learning_rate is None:
            options.learning_rate = 0.01
        if options.initial_point is None:
            options.initial_point = 0
        if options.iterations is None:
            options.iterations = 200
        if options.momentum is None:
            options.momentum = 0.9
    elif str(options.algorithm).upper() == "ADAM":
        if options.learning_rate is None:
            options.learning_rate = 0.1
        if options.initial_point is None:
            options.initial_point = 10
        if options.iterations is None:
            options.iterations = 200
        if options.beta_1 is None:
            options.beta_1 = 0.9
        if options.beta_2 is None:
            options.beta_2 = 0.999
    else:
        raise Exception('Illegal algorithm - choose a valid algorithm [SGD, SGDM, ADAM]')
    return options


def main():

    parser = argparse.ArgumentParser(
        description='Choose an optimising algorithm (-algo / --algorithm), function to optimise '
                    '(-if / --input_function) and (optional) parameters: learning rate (-lr / --learning_rate), '
                    'initial point (-ip / --initial_point), number of iterations (-it / --iterations)'
                    ', momentum (-momentum) for SGD and decays (-b1, -b2) for ADAM: ')
    parser.add_argument('-algo', '--algorithm', type=str, action="store", nargs='?', help="Chosen algorithm",
                        dest="algorithm")
    parser.add_argument('-if', '--input_function', type=str, action="store", nargs='?', help="Function to optimise",
                        dest="input_function")
    parser.add_argument('-lr', '--learning_rate', type=float, action="store", nargs='?',
                        help="Learning rate for the algorithm", dest="learning_rate")
    parser.add_argument('-ip', "--initial_point", type=int, action="store", nargs='?',
                        help="Initial point of the algorithm", dest="initial_point")
    parser.add_argument('-it', '--iterations', type=int, action="store", nargs='?',
                        help="Number of iterations for the algorithm", dest="iterations")
    parser.add_argument('-momentum', '--momentum', type=float, action="store", nargs='?', help="Chosen momentum"
                        , dest="momentum")
    parser.add_argument('-b1', '--beta_1', type=float, action="store", nargs='?', dest="beta_1"
                        , help="Chosen exponential decay for the first moment")
    parser.add_argument('-b2', '--beta_2', type=float, action="store", nargs='?', dest="beta_2"
                        , help="Chosen exponential decay rate for the second moment")

    options = parser.parse_args()
    check_arguments(options)

    function = PolynomialFunction(options.input_function)
    algorithm = options.algorithm.upper()
    learning_rate = options.learning_rate
    initial_point = options.initial_point
    num_of_iterations = options.iterations
    momentum = options.momentum
    beta_1 = options.beta_1
    beta_2 = options.beta_2

    print("Given function: " + str(
        options.input_function) + " Choosen algorithm: " + algorithm + " Learning rate: " + str(
        learning_rate) + " Inital point: " + str(initial_point) + " Number of iterations: " + str(num_of_iterations))
    if algorithm == "SGDM":
        print(" Momentum: " + str(momentum))
    if algorithm == "ADAM":
        print(" Exponential decays for first and second momentum: " + str(beta_1) + ", " + str(beta_2))

    print(function.__call__(opt.optimise(algorithm, function, learning_rate, initial_point, num_of_iterations, True
                                         , momentum, beta_1, beta_2)))


if __name__ == '__main__':
    main()
