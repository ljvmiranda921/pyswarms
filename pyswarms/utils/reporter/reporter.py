# -*- coding: utf-8 -*-
# Import standard library
import os
from typing import Any, Dict, List, Optional

# Import modules
from loguru import logger
from tqdm import tqdm, trange


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
    """

    t: Optional["tqdm[int]"] = None

    def __init__(
        self,
        log_path: Optional[str] = None,
    ):
        """Initialize the reporter

        Attributes
        ----------
        log_path : str, optional
            Sets the default log path (overriden when :code:`path` is given to
            :code:`_setup_logger()`)
        """
        self.log_path = log_path or (os.getcwd() + "/report.log")
        self._bar_fmt = "{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}"
        self._setup_logger()

    def setup_logger(self):
        """Set-up the logger with default values"""
        logger.add(self.log_path, rotation=10485760)

    def pbar(self, iters: int, desc: Optional[str] = None):
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

    def hook(self, *args: List[Any], **kwargs: Dict[str, Any]):
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
        if self.t is None:
            return

        self.t.set_postfix(*args, **kwargs)
