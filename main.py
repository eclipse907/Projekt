#!/usr/bin/env python3

from onevariable_function import OnevariableFunction
import optimisation as opt
import argparse


def check_arguments(options):
    if str(options.algorithm).upper() == "SGD":
        if options.learning_rate is None:
            options.learning_rate = 0.01
        if options.initial_point is None:
            options.initial_point = 0
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
            options.initial_point = 0
        if options.iterations is None:
            options.iterations = 200
        if options.beta1 is None:
            options.beta1 = 0.9
        if options.beta2 is None:
            options.beta2 = 0.999
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
    parser.add_argument('-mom', '--momentum', type=float, action="store", nargs='?', help="Chosen momentum"
                        , dest="momentum")
    parser.add_argument('-b1', '--beta1', type=float, action="store", nargs='?', dest="beta1"
                        , help="Chosen exponential decay for the first moment")
    parser.add_argument('-b2', '--beta2', type=float, action="store", nargs='?', dest="beta2"
                        , help="Chosen exponential decay rate for the second moment")

    options = parser.parse_args()
    check_arguments(options)

    function = OnevariableFunction(options.input_function)
    algorithm = options.algorithm.upper()
    learning_rate = options.learning_rate
    initial_point = options.initial_point
    num_of_iterations = options.iterations
    momentum = options.momentum
    beta1 = options.beta1
    beta2 = options.beta2


    is_to_be_plotted = True

    print("Given function: " + str(
        options.input_function) + " Choosen algorithm: " + algorithm + " Learning rate: " + str(
        learning_rate) + " Inital point: " + str(initial_point) + " Number of iterations: " + str(num_of_iterations))
    if algorithm == "SGDM":
        print(" Momentum: " + str(momentum))
    if algorithm == "ADAM":
        print(" Exponential decays for first and second momentum: " + str(beta1) + ", " + str(beta2))

    print(function.__call__(opt.optimise(algorithm, function, learning_rate, initial_point, num_of_iterations, True
                                         , momentum, beta1, beta2)))


if __name__ == '__main__':
    main()
