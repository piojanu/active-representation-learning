"""Some simple logging functionality, inspired by spinup's logging."""

import atexit
import os.path as osp

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string."""
    num = {
        "gray": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "crimson": 38,
    }[color]
    if highlight:
        num += 10

    attr = [str(num)]
    if bold:
        attr.append("1")

    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


class Logger:
    """A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(self, output_fname="progress.txt", neptune_kwargs=None):
        """Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to.

            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.

            neptune_kwargs (dict): Neptune init kwargs. If None, then Neptune
                logging is disabled.
        """
        self.output_file = open(osp.abspath(output_fname), "w")
        atexit.register(self.output_file.close)

        if neptune_kwargs is not None:
            import neptune.new as neptune

            self.neptune_run = neptune.init(**neptune_kwargs)
        else:
            self.neptune_run = None
        self.writer = SummaryWriter(log_dir=osp.abspath("tb"))

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

    def log_tabular(self, key, val):
        """Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key %s that "
                "you didn't include in the first iteration" % key
            )
        assert key not in self.log_current_row, (
            "You already set %s this iteration. "
            "Maybe you forgot to call dump_tabular()" % key
        )
        self.log_current_row[key] = val

    def dump_tabular(self, global_step):
        """Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.

        Args:
            global_step (int): Global step value of the diagnostics.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            vals.append(val)
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            if val == "":
                continue
            self.writer.add_scalar(key, val, global_step)
            if self.neptune_run is not None:
                self.neptune_run[key].log(val, global_step)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()

        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()
        self.histogram_dict = dict()

    def store(self, **kwargs):
        """Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if k not in self.epoch_dict.keys():
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(
        self,  # pylint: disable=arguments-differ
        key,
        val=None,
        with_min_and_max=False,
        average_only=False,
    ):
        """Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            vals = self.epoch_dict[key]
            self.histogram_dict[key + "/Hist"] = np.array(vals)
            super().log_tabular(key + "/Avg", np.mean(vals))
            if not average_only:
                super().log_tabular(key + "/Std", np.std(vals))
            if with_min_and_max:
                super().log_tabular(key + "/Max", np.max(vals))
                super().log_tabular(key + "/Min", np.min(vals))
        self.epoch_dict[key] = []

    def dump_tabular(self, global_step):
        """Write all of the diagnostics from the current iteration.

        Args:
            global_step (int): Global step value of the diagnostics.
        """
        super().dump_tabular(global_step)
        for key, vals in self.histogram_dict.items():
            self.writer.add_histogram(key, vals, global_step)
        self.histogram_dict.clear()
