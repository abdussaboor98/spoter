"""Leave-one-user-out dataset builder for SPOTER Arabic landmarks.

This utility consumes the CSV landmark files produced by
``data_structurization/arabic_sign_landmarks.py`` and assembles
a leave-one-user-out (LOO) split for a specific signer.  The
resulting train/test CSVs can be fed directly into the SPOTER
training pipeline.

Usage example::

    python data_structurization/create_loo_split.py \
        --landmarks-root /path/to/landmarks \
        --output-dir /path/to/output \
        --user user01

By default the training set will include the original data plus the
``flipped``, ``speed_up`` and ``slow_down`` augmentations for every
user except the held-out one.  Individual augmentations can be
excluded with the ``--no-<augmentation>`` flags.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


AUGMENTATION_DIRS: Dict[str, str] = {
    "original": "original",
    "flipped": "flipped",
    "speed_up": "speed_up",
    "slow_down": "slow_down",
}


@dataclass
class SplitConfig:
    landmarks_root: Path
    output_dir: Path
    target_user: str
    include_flipped: bool
    include_speed_up: bool
    include_slow_down: bool

    @property
    def train_output_path(self) -> Path:
        return self.output_dir / f"train_{self.target_user}.csv"

    @property
    def test_output_path(self) -> Path:
        return self.output_dir / f"test_{self.target_user}.csv"


def _discover_users(original_dir: Path) -> List[str]:
    users = sorted(path.stem for path in original_dir.glob("*.csv") if path.is_file())
    if not users:
        raise FileNotFoundError(
            f"No CSV files found under '{original_dir}'. Did you run the extraction script?"
        )
    return users


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected CSV file not found: {path}")
    return pd.read_csv(path)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_training_frames(
    config: SplitConfig,
    other_users: Sequence[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    # Always include original data for the remaining users.
    original_dir = config.landmarks_root / AUGMENTATION_DIRS["original"]
    for user in other_users:
        user_path = original_dir / f"{user}.csv"
        if not user_path.is_file():
            logging.warning("Skipping missing original CSV for %s", user)
            continue
        frames.append(_load_csv(user_path))

    augmentation_sources: List[tuple[str, bool]] = [
        ("flipped", config.include_flipped),
        ("speed_up", config.include_speed_up),
        ("slow_down", config.include_slow_down),
    ]

    for augmentation, should_include in augmentation_sources:
        if not should_include:
            logging.info("Skipping %s augmentation as requested", augmentation)
            continue

        aug_dir = config.landmarks_root / AUGMENTATION_DIRS[augmentation]
        for user in other_users:
            user_path = aug_dir / f"{user}.csv"
            if not user_path.is_file():
                logging.warning(
                    "Missing %s CSV for %s at %s", augmentation, user, user_path
                )
                continue
            frames.append(_load_csv(user_path))

    if not frames:
        raise RuntimeError("Training set would be empty; no CSV files were loaded.")

    return pd.concat(frames, ignore_index=True)


def _collect_test_frames(config: SplitConfig) -> pd.DataFrame:
    original_dir = config.landmarks_root / AUGMENTATION_DIRS["original"]
    user_path = original_dir / f"{config.target_user}.csv"
    logging.debug("Loading test CSV from %s", user_path)
    return _load_csv(user_path)


def _validate_directories(config: SplitConfig) -> None:
    for name, dirname in AUGMENTATION_DIRS.items():
        candidate = config.landmarks_root / dirname
        if not candidate.is_dir():
            if name == "original":
                raise FileNotFoundError(
                    f"Required directory '{candidate}' does not exist."
                )
            logging.warning(
                "Augmentation directory '%s' does not exist. Optional data will be skipped.",
                candidate,
            )


def build_leave_one_user_out_split(config: SplitConfig) -> None:
    _validate_directories(config)
    _ensure_output_dir(config.output_dir)

    original_dir = config.landmarks_root / AUGMENTATION_DIRS["original"]
    users = _discover_users(original_dir)

    if config.target_user not in users:
        raise ValueError(
            f"Target user '{config.target_user}' does not have an original CSV in {original_dir}."
        )

    other_users = [user for user in users if user != config.target_user]

    if not other_users:
        raise ValueError("No other users available to form the training set.")

    logging.info("Preparing leave-one-user-out split for %s", config.target_user)

    train_df = _collect_training_frames(config, other_users)
    test_df = _collect_test_frames(config)

    logging.info("Writing training CSV to %s", config.train_output_path)
    train_df.to_csv(config.train_output_path, index=False)

    logging.info("Writing test CSV to %s", config.test_output_path)
    test_df.to_csv(config.test_output_path, index=False)

    logging.info(
        "Split generation complete: %d training samples, %d test samples",
        len(train_df),
        len(test_df),
    )


def parse_args(argv: Sequence[str] | None = None) -> SplitConfig:
    parser = argparse.ArgumentParser(
        description="Create leave-one-user-out train/test CSVs from extracted landmarks.",
    )
    parser.add_argument(
        "--landmarks-root",
        type=Path,
        required=True,
        help="Directory containing the 'original', 'flipped', 'speed_up', and 'slow_down' CSV sub-directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the aggregated train/test CSV files will be written.",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="User identifier to hold out (e.g. 'user01').",
    )
    parser.add_argument(
        "--no-flipped",
        dest="include_flipped",
        action="store_false",
        help="Exclude the horizontally flipped augmentation from the training set.",
    )
    parser.add_argument(
        "--no-speed-up",
        dest="include_speed_up",
        action="store_false",
        help="Exclude the speed-up augmentation from the training set.",
    )
    parser.add_argument(
        "--no-slow-down",
        dest="include_slow_down",
        action="store_false",
        help="Exclude the slow-down augmentation from the training set.",
    )
    parser.set_defaults(
        include_flipped=True,
        include_speed_up=True,
        include_slow_down=True,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    return SplitConfig(
        landmarks_root=args.landmarks_root,
        output_dir=args.output_dir,
        target_user=args.user,
        include_flipped=args.include_flipped,
        include_speed_up=args.include_speed_up,
        include_slow_down=args.include_slow_down,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    build_leave_one_user_out_split(config)


if __name__ == "__main__":
    main()
