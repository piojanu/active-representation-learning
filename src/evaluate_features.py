#!/usr/bin/env python3

import argparse
import os.path as osp

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def encode_frames(encoder, frames_nhwc):
    # Transpose to NCHW
    with torch.no_grad():
        frames_nchw = frames_nhwc.transpose(0, 3, 1, 2)
        frames_tensor = torch.from_numpy(frames_nchw).float()
        features = encoder(frames_tensor / 255.0).numpy()

    return features


def train_classifier(features, labels):
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    clf.fit(features, labels)

    return clf


def evaluate_model(model, features, labels):
    return model.score(features, labels)


def main(data_path, ckpt_dir, state_idx):
    # Load the data
    with np.load(data_path) as data:
        train_frames = data["train_frames"]
        train_labels = data["train_labels"]
        test_frames = data["test_frames"]
        test_labels = data["test_labels"]

    # Load the features extractor
    encoder, _, _ = torch.load(
        osp.join(ckpt_dir, "checkpoint.pkl"), map_location=torch.device("cpu")
    )
    encoder_state_dict, _ = torch.load(
        osp.join(ckpt_dir, f"{state_idx}.pt"), map_location=torch.device("cpu")
    )
    encoder.load_state_dict(encoder_state_dict)

    train_features = encode_frames(encoder, train_frames)
    test_features = encode_frames(encoder, test_frames)

    model = train_classifier(train_features, train_labels)

    accuracy = evaluate_model(model, test_features, test_labels)

    print(f"TOP1 accuracy: {accuracy}")


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

    parser = argparse.ArgumentParser(description="Evaluate the learned features.")
    parser.add_argument("--data_path", type=_file_path, help="Path to the dataset.")
    parser.add_argument(
        "--ckpt_dir", type=_dir_path, help="Dir. path to the encoder checkpoints."
    )
    parser.add_argument(
        "--state_idx", type=int, help="Index of the encoder state dict."
    )
    args = parser.parse_args()

    main(args.data_path, args.ckpt_dir, args.state_idx)
