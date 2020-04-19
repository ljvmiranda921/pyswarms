# -*- coding: utf-8 -*-
# Import standard library
import logging
import logging.config
import os
import pprint

# Import modules
import yaml
from tqdm import trange


class Reporter(object):
    """A Reporter object that abstracts various logging capabilities

    To set-up a Reporter, simply perform the following tasks:

    .. code-block:: python

        from pyswarms.utils import Reporter

        rep = Reporter()
        rep.log("Here's my message", lvl=logging.INFO)

    This will set-up a reporter with a default configuration that
    logs to a file, `report.log`, on the current working directory.
    You can change the log path by passing a string to the `log_path`
    parameter:

    .. code-block:: python

        from pyswarms.utils import Reporter

        rep = Reporter(log_path="/path/to/log/file.log")
        rep.log("Here's my message", lvl=logging.INFO)

    If you are working on a module and you have an existing logger,
    you can pass that logger instance during initialization:

    .. code-block:: python

        # mymodule.py
        from pyswarms.utils import Reporter

        # An existing logger in a module
        logger = logging.getLogger(__name__)
        rep = Reporter(logger=logger)

    Lastly, if you have your own logger configuration (YAML file),
    then simply pass that to the `config_path` parameter. This
    overrides the default configuration (including `log_path`):

    .. code-block:: python

        from pyswarms.utils import Reporter

        rep = Reporter(config_path="/path/to/config/file.yml")
        rep.log("Here's my message", lvl=logging.INFO)

    """

    def __init__(
        self, log_path=None, config_path=None, logger=None, printer=None
    ):
        """Initialize the reporter

        Attributes
        ----------
        log_path : str, optional
            Sets the default log path (overriden when :code:`path` is given to
            :code:`_setup_logger()`)
        config_path : str, optional
            Sets the configuration path for custom loggers
        logger : logging.Logger, optional
            The logger object. By default, it creates a new :code:`Logger`
            instance
        printer : pprint.PrettyPrinter, optional
            A printer object. By default, it creates a :code:`PrettyPrinter`
            instance with default values
        """
        self.logger = logger or logging.getLogger(__name__)
        self.printer = printer or pprint.PrettyPrinter()
        self.log_path = log_path or (os.getcwd() + "/report.log")
        self._bar_fmt = "{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}"
        self._env_key = "LOG_CFG"
        self._default_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
                "file_default": {
                    "level": "INFO",
                    "formatter": "standard",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": self.log_path,
                    "encoding": "utf8",
                    "maxBytes": 10485760,
                    "backupCount": 20,
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default", "file_default"],
                    "level": "INFO",
                    "propagate": True,
                }
            },
        }
        self._setup_logger(config_path)

    def log(self, msg, lvl=logging.INFO, *args, **kwargs):
        """Log a message within a set level

        This method abstracts the logging.Logger.log() method. We use this
        method during major state changes, errors, or critical events during
        the optimization run.

        You can check logging levels on this `link`_. In essence, DEBUG is 10,
        INFO is 20, WARNING is 30, ERROR is 40, and CRITICAL is 50.

        .. _link: https://docs.python.org/3/library/logging.html#logging-levels

        Parameters
        ----------
        msg : str
            Message to be logged
        lvl : int, optional
            Logging level. Default is `logging.INFO`
        """
        self.logger.log(lvl, msg, *args, **kwargs)

    def print(self, msg, verbosity, threshold=0):
        """Print a message into console

        This method can be called during non-system calls or minor state
        changes. In practice, we call this method when reporting the cost
        on a given timestep.

        Parameters
        ----------
        msg : str
            Message to be printed
        verbosity : int
            Verbosity parameter, prints message when it's greater than the
            threshold
        threshold : int, optional
            Threshold parameter, prints message when it's lesser than the
            verbosity. Default is `0`
        """
        if verbosity > threshold:
            self.printer.pprint(msg)
        else:
            pass

    def _setup_logger(self, path=None):
        """Set-up the logger with default values

        This method is called right after initializing the Reporter module.
        If no path is supplied, then it loads a default configuration.
        You can view the defaults via the Reporter._default_config attribute.


        Parameters
        ----------
        path : str, optional
            Path to a YAML configuration. If not supplied, uses
            a default config.
        """
        value = path or os.getenv(self._env_key, None)
        try:
            with open(value, "rt") as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        except (TypeError, FileNotFoundError):
            self._load_defaults()

    def _load_defaults(self):
        """Load default logging configuration"""
        logging.config.dictConfig(self._default_config)

    def pbar(self, iters, desc=None):
        """Create a tqdm iterable

        You can use this method to create progress bars. It uses a set
        of abstracted methods from tqdm:

        .. code-block:: python

            from pyswarms.utils import Reporter

            rep = Reporter()
            # Create a progress bar
            for i in rep.pbar(100, name="Optimizer")
                    pass

        Parameters
        ----------
        iters : int
            Maximum range passed to the tqdm instance
        desc : str, optional
            Name of the progress bar that will be displayed

        Returns
        -------
        :obj:`tqdm._tqdm.tqdm`
            A tqdm iterable
        """
        self.t = trange(iters, desc=desc, bar_format=self._bar_fmt)
        return self.t

    def hook(self, *args, **kwargs):
        """Set a hook on the progress bar

        Method for creating a postfix in tqdm. In practice we use this
        to report the best cost found during an iteration:

        .. code-block:: python

            from pyswarms.utils import Reporter

            rep = Reporter()
            # Create a progress bar
            for i in rep.pbar(100, name="Optimizer")
                    best_cost = compute()
                    rep.hook(best_cost=best_cost)
        """
        self.t.set_postfix(*args, **kwargs)
