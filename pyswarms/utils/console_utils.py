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

def end_report(cost, pos, verbosity):
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
    
    # Cuts the length of the best position if it's too long
    if len(list(pos)) > 3:
        out = ('[ ' + 3 * '{:3f} ' + '...]').format(*list(pos))
    else:
        out = list(pos)

    template = ('================================\n'
                'Optimization finished!\n'
                'Final cost: {:06.4f}\n'
                'Best value: {}\n').format(cost, out)
    if verbosity >= 1:
        print(template)