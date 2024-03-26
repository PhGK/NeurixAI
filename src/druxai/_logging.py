import logging
import os

__all__ = ["logger"]


def _setup_logger(log_path="../../logs/") -> "logging.Logger":

    from rich.console import Console
    from rich.logging import RichHandler

    os.makedirs(log_path, exist_ok=True)
    log_file_path = os.path.join(log_path, "logfile.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Console handler
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=False)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # This prevents double outputs
    logger.propagate = False
    return logger


logger = _setup_logger()
