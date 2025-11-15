"""Arabic Sign Language landmark extraction pipeline for SPOTER.

This module implements a command line utility that traverses a dataset of
recorded sign language videos, extracts body and hand landmarks for every
frame using MediaPipe Holistic, and stores the results in CSV files that are
compatible with the SPOTER training pipeline.

The expected input directory structure is the following::

    dataset/
        user01/
            G01/
                R01.mp4
                R02.mp4
                ...
            G02/
            ...
        user02/
        ...

Where ``userXX`` denotes the signer, ``GYY`` the gesture identifier and
``RZZ.mp4`` the repetition index. For every video, this script produces four
CSV rows:

``original``
    Landmarks extracted from the raw video frames.
``flipped``
    Horizontally mirrored landmarks (left/right body parts swapped) to emulate
    both left- and right-handed signing.
``speed_up``
    Temporally down-sampled landmarks approximating a 1.5x speed-up.
``slow_down``
    Temporally up-sampled landmarks approximating a 0.5x slow-down.

Each augmentation is written to a dedicated output sub-directory so that
leave-one-user-out splits can exclude both the original and augmented data of
the held-out signer with a simple directory-level filter.

Example usage::

    python data_structurization/arabic_sign_landmarks.py \
        --dataset-root /path/to/dataset \
        --output-root /path/to/output_dir

The resulting structure will resemble::

    /path/to/output_dir/
        original/user01.csv
        flipped/user01.csv
        speed_up/user01.csv
        slow_down/user01.csv
        original/user02.csv
        ...

The CSV schema matches the one consumed by :mod:`datasets.czech_slr_dataset`
so the generated files can be supplied directly to the SPOTER training
scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from itertools import chain
from multiprocessing import Manager
from queue import Empty

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure repository root is on sys.path when the script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS as HAND_BASE_IDENTIFIERS

# Type aliases for readability
Point = Tuple[float, float]
SequenceDict = Dict[str, List[Point]]
MetadataValue = str | int | float
RowDict = Dict[str, MetadataValue]
UserRows = MutableMapping[str, List[RowDict]]
RowsByAugmentation = Dict[str, UserRows]


# MediaPipe helpers ---------------------------------------------------------

mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


BODY_LANDMARK_MAPPING: Mapping[str, int] = {
    "nose": mp_pose.PoseLandmark.NOSE.value,
    "rightEye": mp_pose.PoseLandmark.RIGHT_EYE.value,
    "leftEye": mp_pose.PoseLandmark.LEFT_EYE.value,
    "rightEar": mp_pose.PoseLandmark.RIGHT_EAR.value,
    "leftEar": mp_pose.PoseLandmark.LEFT_EAR.value,
    "rightShoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "leftShoulder": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "rightElbow": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "leftElbow": mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "rightWrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "leftWrist": mp_pose.PoseLandmark.LEFT_WRIST.value,
}


HAND_LANDMARK_MAPPING: Mapping[str, int] = {
    "wrist": mp_hands.HandLandmark.WRIST.value,
    "thumbCMC": mp_hands.HandLandmark.THUMB_CMC.value,
    "thumbMP": mp_hands.HandLandmark.THUMB_MCP.value,
    "thumbIP": mp_hands.HandLandmark.THUMB_IP.value,
    "thumbTip": mp_hands.HandLandmark.THUMB_TIP.value,
    "indexMCP": mp_hands.HandLandmark.INDEX_FINGER_MCP.value,
    "indexPIP": mp_hands.HandLandmark.INDEX_FINGER_PIP.value,
    "indexDIP": mp_hands.HandLandmark.INDEX_FINGER_DIP.value,
    "indexTip": mp_hands.HandLandmark.INDEX_FINGER_TIP.value,
    "middleMCP": mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value,
    "middlePIP": mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value,
    "middleDIP": mp_hands.HandLandmark.MIDDLE_FINGER_DIP.value,
    "middleTip": mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value,
    "ringMCP": mp_hands.HandLandmark.RING_FINGER_MCP.value,
    "ringPIP": mp_hands.HandLandmark.RING_FINGER_PIP.value,
    "ringDIP": mp_hands.HandLandmark.RING_FINGER_DIP.value,
    "ringTip": mp_hands.HandLandmark.RING_FINGER_TIP.value,
    "littleMCP": mp_hands.HandLandmark.PINKY_MCP.value,
    "littlePIP": mp_hands.HandLandmark.PINKY_PIP.value,
    "littleDIP": mp_hands.HandLandmark.PINKY_DIP.value,
    "littleTip": mp_hands.HandLandmark.PINKY_TIP.value,
}


# Construct ordered identifiers to keep column ordering deterministic.
HAND_IDENTIFIERS: List[str] = list(HAND_BASE_IDENTIFIERS)
ORDERED_IDENTIFIERS: List[str] = (
    list(BODY_IDENTIFIERS)
    + [f"{identifier}_0" for identifier in HAND_IDENTIFIERS]
    + [f"{identifier}_1" for identifier in HAND_IDENTIFIERS]
)


BODY_SWAP_PAIRS: Sequence[Tuple[str, str]] = (
    ("leftEye", "rightEye"),
    ("leftEar", "rightEar"),
    ("leftShoulder", "rightShoulder"),
    ("leftElbow", "rightElbow"),
    ("leftWrist", "rightWrist"),
)


METADATA_COLUMNS: Sequence[str] = (
    "user",
    "gesture",
    "repetition",
    "labels",
    "frame_count",
    "extraction_time_sec",
    "video_path",
)


def _clip_coordinate(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _landmark_or_zero(
    landmark_list: mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None,
    index: int,
    min_visibility: float,
) -> Point:
    if not landmark_list:
        return 0.0, 0.0

    if index >= len(landmark_list.landmark):
        return 0.0, 0.0

    landmark = landmark_list.landmark[index]
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)

    if visibility < min_visibility or presence < min_visibility:
        return 0.0, 0.0

    return _clip_coordinate(landmark.x), _clip_coordinate(landmark.y)


def _extract_pose_landmarks(
    pose_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None,
    min_visibility: float,
) -> Dict[str, Point]:
    pose_points: Dict[str, Point] = {identifier: (0.0, 0.0) for identifier in BODY_IDENTIFIERS}

    for identifier, index in BODY_LANDMARK_MAPPING.items():
        pose_points[identifier] = _landmark_or_zero(pose_landmarks, index, min_visibility)

    # Neck is approximated as the midpoint between both shoulders if available
    left_shoulder = pose_points["leftShoulder"]
    right_shoulder = pose_points["rightShoulder"]
    if left_shoulder != (0.0, 0.0) and right_shoulder != (0.0, 0.0):
        neck = (
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
        )
        pose_points["neck"] = neck
    else:
        pose_points["neck"] = (0.0, 0.0)

    return pose_points


def _extract_hand_landmarks(
    hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None,
) -> Dict[str, Point]:
    hand_points: Dict[str, Point] = {identifier: (0.0, 0.0) for identifier in HAND_IDENTIFIERS}

    if not hand_landmarks:
        return hand_points

    for identifier, index in HAND_LANDMARK_MAPPING.items():
        x = hand_landmarks.landmark[index].x
        y = hand_landmarks.landmark[index].y
        hand_points[identifier] = (_clip_coordinate(x), _clip_coordinate(y))

    return hand_points


def _initialize_sequence_dict() -> SequenceDict:
    sequence: SequenceDict = {identifier: [] for identifier in BODY_IDENTIFIERS}
    for identifier in HAND_IDENTIFIERS:
        sequence[f"{identifier}_0"] = []
        sequence[f"{identifier}_1"] = []
    return sequence


def extract_video_landmarks(
    video_path: Path,
    holistic: mp_holistic.Holistic,
    min_visibility: float,
) -> SequenceDict:
    """Extract landmarks from ``video_path`` using the provided ``holistic`` model."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    sequence = _initialize_sequence_dict()

    while True:
        success, frame = capture.read()
        if not success:
            break

        # MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = holistic.process(frame_rgb)

        pose_points = _extract_pose_landmarks(results.pose_landmarks, min_visibility)
        left_hand_points = _extract_hand_landmarks(results.left_hand_landmarks)
        right_hand_points = _extract_hand_landmarks(results.right_hand_landmarks)

        for identifier in BODY_IDENTIFIERS:
            sequence[identifier].append(pose_points[identifier])

        for identifier in HAND_IDENTIFIERS:
            sequence[f"{identifier}_0"].append(left_hand_points[identifier])
            sequence[f"{identifier}_1"].append(right_hand_points[identifier])

    capture.release()

    return sequence


def _mirror_point(point: Point) -> Point:
    if point == (0.0, 0.0):
        return point
    return (1.0 - point[0], point[1])


def flip_sequence(sequence: SequenceDict) -> SequenceDict:
    """Create a horizontally flipped copy of ``sequence``."""

    flipped = {key: [ _mirror_point(p) for p in frames ] for key, frames in sequence.items()}

    # Swap left/right body parts
    for left_identifier, right_identifier in BODY_SWAP_PAIRS:
        flipped[left_identifier], flipped[right_identifier] = (
            flipped[right_identifier],
            flipped[left_identifier],
        )

    # Swap hands (0 -> left, 1 -> right)
    for identifier in HAND_IDENTIFIERS:
        left_key = f"{identifier}_0"
        right_key = f"{identifier}_1"
        flipped[left_key], flipped[right_key] = flipped[right_key], flipped[left_key]

    return flipped


def _interpolate_pair(values: Sequence[Point], position: float) -> Point:
    if not values:
        return 0.0, 0.0

    lower_index = int(np.floor(position))
    upper_index = min(lower_index + 1, len(values) - 1)
    weight = position - lower_index

    lower_point = values[lower_index]
    upper_point = values[upper_index]

    if lower_index == upper_index or lower_point == upper_point:
        return lower_point

    x = (1 - weight) * lower_point[0] + weight * upper_point[0]
    y = (1 - weight) * lower_point[1] + weight * upper_point[1]
    return (float(x), float(y))


def resample_sequence(sequence: SequenceDict, speed_factor: float) -> SequenceDict:
    """Resample ``sequence`` to emulate a temporal speed change."""

    if speed_factor <= 0:
        raise ValueError("speed_factor must be greater than zero")

    if not sequence:
        return {}

    any_key = next(iter(sequence))
    original_length = len(sequence[any_key])

    if original_length == 0:
        return {key: [] for key in sequence}

    if np.isclose(speed_factor, 1.0):
        return {key: list(frames) for key, frames in sequence.items()}

    target_length = max(1, int(round(original_length / speed_factor)))
    # Ensure that the final frame is always included in the resampled sequence
    positions = np.linspace(0, max(original_length - 1, 0), target_length)

    resampled: SequenceDict = {key: [] for key in sequence}

    for identifier, frames in sequence.items():
        for position in positions:
            resampled[identifier].append(_interpolate_pair(frames, float(position)))

    return resampled


def sequence_to_row(sequence: SequenceDict, metadata: Mapping[str, MetadataValue]) -> RowDict:
    row: RowDict = dict(metadata)

    if sequence:
        example_key = next(iter(sequence))
        row["frame_count"] = len(sequence[example_key])
    else:
        row["frame_count"] = 0

    for identifier in ORDERED_IDENTIFIERS:
        frames = sequence[identifier]
        xs = [round(point[0], 6) for point in frames]
        ys = [round(point[1], 6) for point in frames]
        row[f"{identifier}_X"] = json.dumps(xs)
        row[f"{identifier}_Y"] = json.dumps(ys)

    return row


@dataclass
class ExtractionConfig:
    dataset_root: Path
    output_root: Path
    min_detection_confidence: float
    min_tracking_confidence: float
    min_visibility: float
    speed_up_factor: float
    slow_down_factor: float
    workers: int


def discover_videos(dataset_root: Path) -> List[Path]:
    video_paths = sorted(dataset_root.glob("user*/G*/*.mp4"))
    return [path for path in video_paths if path.is_file()]


def _metadata_from_path(dataset_root: Path, video_path: Path) -> Dict[str, MetadataValue]:
    relative_path = video_path.relative_to(dataset_root)
    try:
        user_id, gesture_id, video_name = relative_path.parts
    except ValueError as exc:
        raise ValueError(
            f"Unexpected directory layout for video '{video_path}'."
        ) from exc

    repetition_id = Path(video_name).stem
    label = int(gesture_id[1:])

    return {
        "user": user_id,
        "gesture": gesture_id,
        "repetition": repetition_id,
        "labels": label,
        "video_path": str(relative_path),
    }


def group_videos_by_user(dataset_root: Path, video_paths: Sequence[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in video_paths:
        relative_path = path.relative_to(dataset_root)
        try:
            user_id = relative_path.parts[0]
        except IndexError as exc:
            raise ValueError(f"Unexpected directory layout for video '{path}'.") from exc
        grouped[user_id].append(path)
    return dict(grouped)


def ensure_output_directories(output_root: Path) -> Mapping[str, Path]:
    subdirs = {
        "original": output_root / "original",
        "flipped": output_root / "flipped",
        "speed_up": output_root / "speed_up",
        "slow_down": output_root / "slow_down",
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return subdirs


def _initialize_rows_container() -> RowsByAugmentation:
    return {
        "original": defaultdict(list),
        "flipped": defaultdict(list),
        "speed_up": defaultdict(list),
        "slow_down": defaultdict(list),
    }


def _append_rows_for_sequences(
    rows_by_augmentation: RowsByAugmentation,
    metadata: Mapping[str, MetadataValue],
    base_sequence: SequenceDict,
    speed_up_factor: float,
    slow_down_factor: float,
) -> None:
    user_key = str(metadata["user"])

    rows_by_augmentation["original"][user_key].append(sequence_to_row(base_sequence, metadata))

    flipped_sequence = flip_sequence(base_sequence)
    rows_by_augmentation["flipped"][user_key].append(sequence_to_row(flipped_sequence, metadata))

    speed_up_sequence = resample_sequence(base_sequence, speed_up_factor)
    rows_by_augmentation["speed_up"][user_key].append(sequence_to_row(speed_up_sequence, metadata))

    slow_down_sequence = resample_sequence(base_sequence, slow_down_factor)
    rows_by_augmentation["slow_down"][user_key].append(sequence_to_row(slow_down_sequence, metadata))


def _finalize_rows(rows_by_augmentation: RowsByAugmentation) -> Dict[str, Dict[str, List[RowDict]]]:
    return {
        augmentation: {user: list(rows) for user, rows in user_rows.items()}
        for augmentation, user_rows in rows_by_augmentation.items()
    }


def _merge_rows(
    destination: RowsByAugmentation,
    addition: Mapping[str, Mapping[str, List[RowDict]]],
) -> None:
    for augmentation, user_rows in addition.items():
        destination_user_rows = destination[augmentation]
        for user, rows in user_rows.items():
            destination_user_rows[user].extend(rows)


def _chunk_video_paths(video_paths: Sequence[Path], workers: int) -> List[List[Path]]:
    if workers <= 1 or not video_paths:
        return [list(video_paths)]

    max_chunks = min(len(video_paths), workers)
    array = np.array(video_paths, dtype=object)
    return [list(chunk) for chunk in np.array_split(array, max_chunks) if len(chunk)]


def _chunk_user_groups(
    user_video_map: Mapping[str, Sequence[Path]],
    workers: int,
) -> List[List[Path]]:
    if workers <= 1 or not user_video_map:
        flat = list(chain.from_iterable(user_video_map.values()))
        return [flat] if flat else []

    users = sorted(user_video_map.keys())
    max_chunks = min(len(users), workers)
    user_array = np.array(users, dtype=object)
    chunks: List[List[Path]] = []
    for user_chunk in np.array_split(user_array, max_chunks):
        paths: List[Path] = []
        for user in user_chunk:
            paths.extend(user_video_map[str(user)])
        if paths:
            chunks.append(paths)
    return chunks


def _process_video_batch(
    video_paths: Sequence[str],
    dataset_root: str,
    min_visibility: float,
    speed_up_factor: float,
    slow_down_factor: float,
    holistic_kwargs: Mapping[str, float | bool],
    worker_id: int | None = None,
    progress_queue: Any | None = None,
) -> Dict[str, Dict[str, List[RowDict]]]:
    rows = _initialize_rows_container()
    dataset_root_path = Path(dataset_root)
    with mp_holistic.Holistic(**holistic_kwargs) as holistic:
        for video_path_str in video_paths:
            video_path = Path(video_path_str)
            metadata = _metadata_from_path(dataset_root_path, video_path)
            start_time = time.perf_counter()
            base_sequence = extract_video_landmarks(video_path, holistic, min_visibility)
            metadata = dict(metadata)
            metadata["extraction_time_sec"] = round(time.perf_counter() - start_time, 6)
            _append_rows_for_sequences(
                rows,
                metadata,
                base_sequence,
                speed_up_factor,
                slow_down_factor,
            )
            if progress_queue is not None and worker_id is not None:
                progress_queue.put(worker_id)
    return _finalize_rows(rows)


def write_augmented_csvs(
    rows_by_augmentation: Mapping[str, Mapping[str, List[RowDict]]],
    output_dirs: Mapping[str, Path],
) -> None:
    ordered_columns: List[str] = list(METADATA_COLUMNS)
    for identifier in ORDERED_IDENTIFIERS:
        ordered_columns.append(f"{identifier}_X")
        ordered_columns.append(f"{identifier}_Y")

    for augmentation, users_rows in rows_by_augmentation.items():
        output_dir = output_dirs[augmentation]
        for user, rows in users_rows.items():
            if not rows:
                continue
            dataframe = pd.DataFrame(rows)
            dataframe = dataframe.reindex(columns=ordered_columns)
            output_path = output_dir / f"{user}.csv"
            dataframe.to_csv(output_path, index=False)


def run_extraction(config: ExtractionConfig) -> None:
    dataset_root = config.dataset_root
    video_paths = discover_videos(dataset_root)

    if not video_paths:
        logging.warning("No videos found under %s", dataset_root)
        return

    output_dirs = ensure_output_directories(config.output_root)

    rows_by_augmentation = _initialize_rows_container()
    user_video_map = group_videos_by_user(dataset_root, video_paths)

    holistic_kwargs = {
        "static_image_mode": False,
        "model_complexity": 2,
        "enable_segmentation": False,
        "refine_face_landmarks": False,
        "min_detection_confidence": config.min_detection_confidence,
        "min_tracking_confidence": config.min_tracking_confidence,
    }

    if config.workers <= 1:
        with mp_holistic.Holistic(**holistic_kwargs) as holistic:
            for video_path in tqdm(video_paths, desc="Extracting landmarks"):
                metadata = _metadata_from_path(dataset_root, video_path)

                start_time = time.perf_counter()
                base_sequence = extract_video_landmarks(video_path, holistic, config.min_visibility)
                metadata["extraction_time_sec"] = round(time.perf_counter() - start_time, 6)

                _append_rows_for_sequences(
                    rows_by_augmentation,
                    metadata,
                    base_sequence,
                    config.speed_up_factor,
                    config.slow_down_factor,
                )
    else:
        chunks = _chunk_user_groups(user_video_map, config.workers)
        with Manager() as manager:
            progress_queue = manager.Queue()
            worker_bars: Dict[int, tqdm] = {}
            future_to_worker: Dict[Any, int] = {}
            with tqdm(total=len(video_paths), desc="Total", position=0, leave=True) as total_bar:
                with ProcessPoolExecutor(max_workers=config.workers) as executor:
                    for index, chunk in enumerate(chunks):
                        if not chunk:
                            continue
                        chunk_paths = [str(path) for path in chunk]
                        worker_id = index
                        worker_bars[worker_id] = tqdm(
                            total=len(chunk_paths),
                            desc=f"Worker {index + 1}",
                            position=index + 1,
                            leave=False,
                        )
                        future = executor.submit(
                            _process_video_batch,
                            chunk_paths,
                            str(dataset_root),
                            config.min_visibility,
                            config.speed_up_factor,
                            config.slow_down_factor,
                            holistic_kwargs,
                            worker_id,
                            progress_queue,
                        )
                        future_to_worker[future] = worker_id

                    pending = set(future_to_worker)
                    while pending:
                        try:
                            worker_id = progress_queue.get(timeout=0.1)
                        except Empty:
                            pass
                        else:
                            if worker_id in worker_bars:
                                worker_bars[worker_id].update(1)
                            total_bar.update(1)

                        finished = [future for future in pending if future.done()]
                        for future in finished:
                            chunk_rows = future.result()
                            _merge_rows(rows_by_augmentation, chunk_rows)
                            pending.remove(future)

                    while True:
                        try:
                            worker_id = progress_queue.get_nowait()
                        except Empty:
                            break
                        else:
                            if worker_id in worker_bars:
                                worker_bars[worker_id].update(1)
                            total_bar.update(1)

            for bar in worker_bars.values():
                bar.close()

    write_augmented_csvs(_finalize_rows(rows_by_augmentation), output_dirs)


def parse_arguments() -> ExtractionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the root directory containing the userXX/GYY/RZZ.mp4 structure.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where the CSV files will be written.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe Holistic (default: 0.5).",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence for MediaPipe Holistic (default: 0.5).",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.2,
        help="Minimum landmark visibility required to keep a pose landmark (default: 0.5).",
    )
    parser.add_argument(
        "--speed-up-factor",
        type=float,
        default=1.5,
        help="Temporal resampling factor for the speed-up augmentation (default: 1.5).",
    )
    parser.add_argument(
        "--slow-down-factor",
        type=float,
        default=0.5,
        help="Temporal resampling factor for the slow-down augmentation (default: 0.5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes to use for extraction (default: 1).",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser()

    return ExtractionConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        min_visibility=args.min_visibility,
        speed_up_factor=args.speed_up_factor,
        slow_down_factor=args.slow_down_factor,
        workers=max(1, args.workers),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = parse_arguments()
    run_extraction(config)


if __name__ == "__main__":
    main()
