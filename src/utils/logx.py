"""Some simple logging functionality, inspired by spinup's logging."""
import re

import matplotlib.pyplot as plt
import numpy as np
from aim import Distribution, Image, Run
from torch.utils.tensorboard._utils import convert_to_HWC, figure_to_image


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


class AimRun:
    """Adapter around aim.Run to handle SpinUp-like logging."""

    def __init__(self, cfg):
        self.run = Run(
            repo=cfg._run.scratch_space,
            experiment=cfg._run.exp_name,
            run_hash=cfg._run.run_name,
            log_system_params=True,
        )
        self.run["hparams"] = cfg

    @staticmethod
    def parse_key(key):
        """Parse SpinUp-like key into Aim key and context."""
        name, _, info = key.partition("/")

        if info == "":
            context = dict()
        elif info in ("Avg", "Max", "Min", "Std"):
            context = dict(metric=info)
        elif re.fullmatch("E[0-9]+", info) is not None:
            context = dict(encoder=info[1:])
        else:
            raise NotImplementedError(f"Unknown context: {info}")

        if name.startswith("Train"):
            name = name[5:]
            context["type"] = "Train"
        elif name.startswith("Test"):
            name = name[4:]
            context["type"] = "Test"

        return name, context

    def track(self, key, value, step=None):
        """Track a value."""
        name, context = AimRun.parse_key(key)
        self.run.track(value, name, step=step, context=context)

    def track_histogram(self, key, histogram, step=None):
        """Track a histogram."""
        self.track(key, Distribution(histogram), step=step)

    def track_image(self, key, image, step=None):
        """Track an image.

        Note: We assume the PyTorch image data format (channel first).
        """
        if image.ndim == 4:
            image = convert_to_HWC(image, input_format="NCHW")
        elif image.ndim == 3:
            image = convert_to_HWC(image, input_format="CHW")

        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)

        self.track(key, Image(image), step=step)

    def track_scalar(self, key, scalar, step=None):
        """Track a scalar."""
        self.track(key, scalar, step=step)


class Logger:
    """A general-purpose logger.

    Makes it easy to log diagnostics to CMD and AimStack.
    """

    def __init__(self, cfg):
        """Initialize a Logger."""
        self.aim_run = AimRun(cfg)

        self.log_current_hist = {}
        self.log_current_img = {}
        self.log_current_row = {}

    def log_heatmap(self, key, val):
        """Generate a heatmap and log it as an image."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        ax.imshow(val, cmap="viridis")
        fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax)

        img_arr = figure_to_image(fig)
        self.log_image(key, img_arr)

    def log_histogram(self, key, val):
        """Log a histogram."""

        assert key not in self.log_current_hist, (
            "You already set %s this iteration. "
            "Maybe you forgot to call dump_tabular()" % key
        )
        self.log_current_hist[key] = val

    def log_image(self, key, val):
        """Log an image."""

        assert key not in self.log_current_img, (
            "You already set %s this iteration. "
            "Maybe you forgot to call dump_tabular()" % key
        )
        self.log_current_img[key] = val

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

    def log_video(self, key, val):
        """Log a video."""

        raise NotImplementedError()

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

            # Write to Aim
            self.aim_run.track_scalar(key, val, step=global_step)
        print("-" * n_slashes, flush=True)

        for key, hist in self.log_current_hist.items():
            self.aim_run.track_histogram(key, hist, step=global_step)

        for key, img in self.log_current_img.items():
            self.aim_run.track_image(key, img, step=global_step)

        self.log_current_hist.clear()
        self.log_current_img.clear()
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

            average_only (bool): If true, do not log the std. dev. and histogram
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            vals = self.epoch_dict[key]
            super().log_tabular(key + "/Avg", np.mean(vals))
            if not average_only:
                super().log_histogram(key, np.array(vals))
                super().log_tabular(key + "/Std", np.std(vals))
            if with_min_and_max:
                super().log_tabular(key + "/Max", np.max(vals))
                super().log_tabular(key + "/Min", np.min(vals))
        self.epoch_dict[key] = []
