"""Some simple logging functionality, inspired by spinup's logging."""

import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import tile_images
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

    Makes it easy to log diagnostics to CMD, TensorBoard, or Neptune.
    """

    def __init__(self, neptune_kwargs=None):
        """Initialize a Logger.

        Args:
            neptune_kwargs (dict): Neptune init kwargs. If None, then Neptune
                logging is disabled.
        """
        if neptune_kwargs is not None:
            import neptune.new as neptune

            self.neptune_run = neptune.init(**neptune_kwargs)
        else:
            self.neptune_run = None

        self.writer = SummaryWriter(log_dir=osp.abspath("tb"))

        self.log_current_map = {}
        self.log_current_row = {}

    def log_heatmap(self, key, val):
        """Log a heatmap of some diagnostic."""

        assert key not in self.log_current_map, (
            "You already set %s this iteration. "
            "Maybe you forgot to call dump_tabular()" % key
        )
        self.log_current_map[key] = val

    def log_tabular(self, key, val):
        """Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        assert key not in self.log_current_row, (
            "You already set %s this iteration. "
            "Maybe you forgot to call dump_tabular()" % key
        )
        self.log_current_row[key] = val

    def dump_tabular(self, global_step):
        """Write all of the diagnostics from the current iteration.

        Args:
            global_step (int): Global step value of the diagnostics.
        """
        max_key_len = max(15, *[len(key) for key in self.log_current_row.keys()])
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len

        print("-" * n_slashes)
        for key, val in self.log_current_row.items():
            # Write to CMD
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))

            # Write to TensorBoard
            self.writer.add_scalar(key, val, global_step)

            # Write to Neptune
            if self.neptune_run is not None:
                self.neptune_run[key].log(val, global_step)
        print("-" * n_slashes, flush=True)

        for key, val in self.log_current_map.items():
            # Generate a heatmap
            fig, ax = plt.subplots()
            ax.imshow(val, cmap="viridis")
            fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax)

            self.writer.add_figure(key, fig, global_step)

        self.log_current_map.clear()
        self.log_current_row.clear()


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

    def log_heatmap(self, key, val=None):
        """Log a heatmap or possibly the tiled heatmaps.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: Values for the heatmap. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
        """
        if val is not None:
            super().log_heatmap(key, val)
        else:
            # Add channel dimension
            vals = np.expand_dims(np.asarray(self.epoch_dict[key]), axis=-1)
            tiled_vals = tile_images(vals)
            super().log_heatmap(key, tiled_vals)
        self.epoch_dict[key] = []

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
