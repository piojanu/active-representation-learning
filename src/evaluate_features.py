#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.tensorboard import SummaryWriter

MAX_WORKERS = 8


def my_pid():
    """Returns relative PID of a pool process."""
    return mp.current_process()._identity[0]


def linquad_range(delta, max_value):
    """Range in [0, `max_value`], stepping linearly up to `delta`, then quadratically."""
    idx = 0
    while True:
        if idx < delta:
            value = idx
        else:
            value = (idx - delta) ** 2 + delta

        if value < max_value:
            yield value
            idx += 1
        else:
            yield max_value
            break


def encode_frames(encoder, frames_tensor):
    with torch.no_grad():
        return encoder(frames_tensor).cpu().numpy()


def train_classifier(features, labels):
    clf = make_pipeline(StandardScaler(), LinearSVC())
    clf.fit(features, labels)

    return clf


def evaluate_model(model, features, labels):
    return model.score(features, labels)


def init_worker(data_path, delta_):
    # NOTE: Without limiting threads number the CPU is the bottleneck
    torch.set_num_threads(2)

    global delta
    delta = delta_

    def _get_device_name(rank):
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                return "cuda:0"

            # Distribute remaining environments evenly across GPUs
            return "cuda:" + str(rank % torch.cuda.device_count())
        else:
            return "cpu"

    global device
    device = torch.device(_get_device_name(my_pid()))

    # Load and preprocess the data
    with np.load(data_path) as data:
        global train_frames
        train_frames = (
            # Transpose to NCHW
            torch.from_numpy(data["train_frames"].transpose(0, 3, 1, 2)).to(
                device, dtype=torch.float, non_blocking=True
            )
            / 255.0
        )

        global train_labels
        train_labels = data["train_labels"]

        global test_frames
        test_frames = (
            # Transpose to NCHW
            torch.from_numpy(data["test_frames"].transpose(0, 3, 1, 2)).to(
                device, dtype=torch.float, non_blocking=True
            )
            / 255.0
        )

        global test_labels
        test_labels = data["test_labels"]


def worker(key_n_encoder_dir):
    key, encoder_dir = key_n_encoder_dir

    global delta
    global device
    global train_frames
    global train_labels
    global test_frames
    global test_labels

    encoder, _, _ = torch.load(
        osp.join(encoder_dir, "checkpoint.pkl"), map_location=device
    )

    ckpt_paths = sorted(
        [
            # (<step>, <path>)
            (int(filename.rsplit(".", 1)[0]), osp.join(encoder_dir, filename))
            for filename in os.listdir(encoder_dir)
            if filename.endswith(".pt")
        ],
        key=lambda x: x[0],
    )

    accuracies = []
    for ptr in linquad_range(delta, len(ckpt_paths) - 1):
        step, path = ckpt_paths[ptr]

        # The encoder state dict is at the position zero
        encoder.load_state_dict(torch.load(path, map_location=device)[0])

        train_features = encode_frames(encoder, train_frames)
        test_features = encode_frames(encoder, test_frames)

        model = train_classifier(train_features, train_labels)

        value = evaluate_model(model, test_features, test_labels)
        accuracies.append((step, value))

    return (key, accuracies)


def main(data_path, exp_dir, delta):
    tb_writer = None
    for root, dirs, _ in os.walk(exp_dir):
        if "tb" in dirs:
            print(f"Processing: {osp.relpath(root, start=exp_dir)}")
            tb_writer = SummaryWriter(log_dir=osp.join(root, "tb"))

        if "encoder_0" in dirs:
            encoder_dirs = [
                # (<key>, <path>)
                (f"E{dirname[8:]}", osp.join(root, dirname))
                for dirname in dirs
                if "encoder_" in dirname
            ]
            with mp.Pool(
                processes=MAX_WORKERS,
                initializer=init_worker,
                initargs=[data_path, delta],
            ) as pool:
                result = pool.map(worker, encoder_dirs)
                for key, accuracies in result:
                    # Log accuracies
                    # NOTE: They need to be sorted by step
                    for step, value in accuracies:
                        tb_writer.add_scalar(f"Accuracy/{key}", value, step)


if __name__ == "__main__":

    def _dir_path(path):
        if osp.isdir(path):
            return path
        else:
            raise NotADirectoryError(path)

    def _file_path(path):
        if osp.isfile(path):
            return path
        else:
            raise FileNotFoundError(path)

    parser = argparse.ArgumentParser(description="Evaluate the learned features")
    parser.add_argument("--data_path", type=_file_path, help="Path to the dataset")
    parser.add_argument("--exp_dir", type=_dir_path, help="Dir. path to the experiment")
    parser.add_argument(
        "--delta",
        type=int,
        default=4,
        help="Delta for the linear-quadratic range of checkpoints to evaluate",
    )
    args = parser.parse_args()

    main(args.data_path, args.exp_dir, args.delta)
