# -*- coding: utf-8 -*-

""" console_utils.py: various tools for printing into console """

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules


def cli_print(message, verbosity, threshold, logger):
    """Helper function to print console output

    Parameters
    ----------
    message : str
        the message to be printed into the console
    verbosity : int
        verbosity setting of the user
    threshold : int
        threshold for printing
    logger : logging.getLogger
        logger instance

    """
    if verbosity >= threshold:
        logger.info(message)
    else:
        pass


def end_report(cost, pos, verbosity, logger):
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
    logger : logging.getLogger
        logger instance
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
        logger.info(template)
