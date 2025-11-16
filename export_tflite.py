"""Utility script for exporting a trained SPOTER TCN to a float16 TFLite model."""

import argparse
from pathlib import Path

import tensorflow as tf

from spoter.spoter_model import SPOTER
from spoter.config import DEFAULT_TCN_CHANNELS, parse_tcn_channels, resolve_tcn_channels
from datasets.sign_pose_dataset import ALL_IDENTIFIERS


def _tcn_channels_argument(value: str):
    try:
        return parse_tcn_channels(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a trained SPOTER Temporal Convolutional Network to a float16 TFLite model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained Keras weights file (e.g. checkpoint_v_0.weights.h5).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="out-checkpoints/spoter_float16.tflite",
        help="Where the converted TFLite model should be saved.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of gesture classes the model predicts. Must match the trained checkpoint.",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=len(ALL_IDENTIFIERS),
        help="Number of pose landmarks per frame (equals joints count).",
    )
    parser.add_argument(
        "--tcn_channels",
        type=_tcn_channels_argument,
        default=None,
        help=f"Comma-separated TemporalConvBlock widths (default: {','.join(map(str, DEFAULT_TCN_CHANNELS))}).",
    )
    parser.add_argument(
        "--tcn_dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied inside each TemporalConvBlock.",
    )
    parser.add_argument(
        "--tcn_activation",
        type=str,
        default="relu",
        help="Activation used after input projection. Should match the training configuration.",
    )
    parser.add_argument(
        "--tcn_kernel_size",
        type=int,
        default=5,
        help="Temporal convolution kernel size used by the TemporalConvBlock layers.",
    )
    parser.add_argument(
        "--tcn_dilation_base",
        type=int,
        default=2,
        help="Base of the dilation progression across TemporalConvBlocks.",
    )
    parser.add_argument(
        "--dummy_sequence_length",
        type=int,
        default=128,
        help="Sequence length used to build the model before conversion. Can be any positive value.",
    )
    return parser.parse_args()


def _instantiate_model(args: argparse.Namespace) -> SPOTER:
    tcn_channels = resolve_tcn_channels(args.tcn_channels)
    model = SPOTER(
        num_classes=args.num_classes,
        tcn_channels=tcn_channels,
        dropout=args.tcn_dropout,
        activation=args.tcn_activation,
        kernel_size=args.tcn_kernel_size,
        dilation_base=args.tcn_dilation_base,
    )

    dummy_input = tf.zeros(
        shape=(1, args.dummy_sequence_length, args.num_joints, 2),
        dtype=tf.float32,
    )
    model(dummy_input, training=False)
    model.load_weights(args.checkpoint_path)
    return model


def _convert_to_tflite(model: SPOTER, args: argparse.Namespace) -> bytes:
    inference_function = tf.function(lambda pose: model(pose, training=False))
    concrete_function = inference_function.get_concrete_function(
        tf.TensorSpec(shape=[None, None, args.num_joints, 2], dtype=tf.float32)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    return converter.convert()


def main():
    args = _parse_arguments()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")

    model = _instantiate_model(args)
    tflite_buffer = _convert_to_tflite(model, args)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_buffer)
    print(f"Float16 TFLite model saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
