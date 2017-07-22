# -*- coding: utf-8 -*-

""" console_utils.py: various tools for printing into console """

def cli_print(message, verbosity, threshold):
    """Helper function to print console output

    Parameters
    ----------
    message : str
        the message to be printed into the console
    verbosity : int
        verbosity setting of the user
    threshold : int
        threshold for printing

    """
    if verbosity >= threshold:
        print(message)
    else:
        pass

def end_report(cost, pos, verbosity)
    """Helper function to print a simple report at the end of the
    run. This always has a threshold of 1.

    Parameters
    ----------
    cost : float
        final cost from the optimization procedure.
    pos : numpy.ndarray or list
        best position found
    verbosity : int
        verbosity setting of the user.

    """
    if verbosity >= 1:
        print('================================\n \
               Optimization finished!\n \
               Final cost: %.3f \n \
               Values: %s\n' % (cost, list(pos)))