import os
import argparse
import random
import logging
import json
import math
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.auto import tqdm

from utils import select_training_subset, user_gesture_repetition_validation_split
from datasets.sign_pose_dataset import SignPoseDataset
from spoter.spoter_model import SPOTER
from spoter.utils import train_single_epoch, evaluate_model, PlateauLearningRateScheduler
from spoter.gaussian_noise import AdditiveGaussianNoise
from spoter.config import DEFAULT_TCN_CHANNELS, parse_tcn_channels, resolve_tcn_channels


def _tcn_channels_argument(value: str):
    try:
        return parse_tcn_channels(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_training_arg_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes to be recognized by the model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument(
        "--tcn_channels",
        type=_tcn_channels_argument,
        default=None,
        help=f"Comma-separated TemporalConvBlock widths (default: {','.join(map(str, DEFAULT_TCN_CHANNELS))}).",
    )
    parser.add_argument(
        "--tcn_kernel_size",
        type=int,
        default=5,
        help="Kernel size used for every TemporalConvBlock convolution.",
    )
    parser.add_argument(
        "--tcn_dropout",
        type=float,
        default=0.1,
        help="Dropout rate applied inside each TemporalConvBlock.",
    )
    parser.add_argument(
        "--tcn_activation",
        type=str,
        default="relu",
        help="Activation applied after the frame projection layer.",
    )
    parser.add_argument(
        "--tcn_dilation_base",
        type=int,
        default=2,
        help="Base of the exponential dilation progression across TemporalConvBlocks.",
    )

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="split-from-train", help="Type of validation set construction. See README for further reference")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adamw"],
        default="sgd",
        help="Optimizer used for training.",
    )
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=0.9,
        help="Momentum term for SGD optimizer.",
    )
    parser.add_argument(
        "--adamw_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay applied by the AdamW optimizer.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used for training and evaluation dataloaders")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Number of epochs without validation improvement before stopping early. Set to 0 to disable.")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help="Factor for the PlateauLearningRateScheduler scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the PlateauLearningRateScheduler scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=float, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=float, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    return parser


def _configure_logging(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )


def _load_training_state(state_path: Path) -> dict:
    if not state_path.is_file():
        return {}
    with open(state_path, "r", encoding="utf-8") as state_file:
        return json.load(state_file)


def _save_training_state(state_path: Path, state: dict):
    with open(state_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file)


def _set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _initialize_model(args: argparse.Namespace, sample_shape=None) -> SPOTER:
    tcn_channels = resolve_tcn_channels(args.tcn_channels)
    model_kwargs = dict(
        num_classes=args.num_classes,
        tcn_channels=tcn_channels,
        dropout=args.tcn_dropout,
        activation=args.tcn_activation,
        kernel_size=args.tcn_kernel_size,
        dilation_base=args.tcn_dilation_base,
    )

    model = SPOTER(**model_kwargs)
    return model


def _build_optimizer(args: argparse.Namespace) -> tf.keras.optimizers.Optimizer:
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.sgd_momentum)
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.adamw_weight_decay)
    raise ValueError(f"Unsupported optimizer '{args.optimizer}'.")


def run_training(args):
    _set_random_seeds(args.seed)
    _configure_logging(args)
    args.tcn_channels = resolve_tcn_channels(args.tcn_channels)

    accelerator_type = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"Using {accelerator_type} for training.")

    checkpoint_dir = Path("out-checkpoints") / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_dir / "training_state.json"
    latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.weights.h5"
    training_state = _load_training_state(state_path)

    gaussian_noise = AdditiveGaussianNoise(args.gaussian_mean, args.gaussian_std)
    training_dataset = SignPoseDataset(args.training_set_path)

    if args.validation_set == "from-file":
        validation_dataset = SignPoseDataset(args.validation_set_path)
    elif args.validation_set == "split-from-train":
        training_dataset, validation_dataset = user_gesture_repetition_validation_split(
            training_dataset,
            args.training_set_path,
            seed=args.seed,
        )
    else:
        validation_dataset = None

    if args.testing_set_path:
        test_dataset = SignPoseDataset(args.testing_set_path)
    else:
        test_dataset = None

    if args.experimental_train_split:
        training_dataset = select_training_subset(training_dataset, args.experimental_train_split)

    train_tf_dataset = training_dataset.as_tf_dataset(
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        gaussian_noise=gaussian_noise,
    )

    if validation_dataset:
        validation_tf_dataset = validation_dataset.as_tf_dataset(
            batch_size=1,
            shuffle=False,
            augment=False,
            pad_to_max_length=False,
            return_mask=False,
        )
    else:
        validation_tf_dataset = None

    has_validation = validation_tf_dataset is not None

    if test_dataset:
        test_tf_dataset = test_dataset.as_tf_dataset(
            batch_size=1,
            shuffle=False,
            augment=False,
            pad_to_max_length=False,
            return_mask=False,
        )
    else:
        test_tf_dataset = None

    sample_shape = training_dataset.input_shape

    spoter_model = _initialize_model(args, sample_shape)

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = _build_optimizer(args)
    scheduler = PlateauLearningRateScheduler(optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)

    dummy_input = tf.zeros((1,) + sample_shape, dtype=tf.float32)
    spoter_model(dummy_input, training=False)

    epoch_losses = list(training_state.get("epoch_losses", []))
    training_accuracies = list(training_state.get("training_accuracies", []))
    validation_accuracies = list(training_state.get("validation_accuracies", []))
    learning_rate_history = list(training_state.get("learning_rate_history", []))
    best_training_accuracy = float(training_state.get("best_training_accuracy", 0))
    best_validation_accuracy = float(training_state.get("best_validation_accuracy", 0))
    best_validation_loss = float(training_state.get("best_validation_loss", math.inf))
    checkpoint_rotation_index = int(training_state.get("checkpoint_rotation_index", 0))
    epochs_without_improvement = int(training_state.get("epochs_without_improvement", 0))
    start_epoch = int(training_state.get("next_epoch", 0))
    training_accuracy = training_accuracies[-1] if training_accuracies else 0
    validation_accuracy = validation_accuracies[-1] if validation_accuracies else 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    if latest_checkpoint_path.exists() and start_epoch > 0:
        spoter_model.load_weights(str(latest_checkpoint_path))
        if training_state.get("scheduler_state"):
            scheduler.set_state(training_state["scheduler_state"])
        if "current_learning_rate" in training_state:
            optimizer.learning_rate.assign(tf.cast(training_state["current_learning_rate"], tf.float32))
        resume_message = f"Resuming {args.experiment_name} from epoch {start_epoch + 1}"
        print(resume_message)
        logging.info(resume_message)

    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="Epochs", unit="epoch")
    early_stop_epoch = None
    best_checkpoint_path = checkpoint_dir / "checkpoint_best.weights.h5"
    for epoch in epoch_bar:
        epoch_loss, _, _, training_accuracy = train_single_epoch(
            spoter_model,
            train_tf_dataset,
            loss_function,
            optimizer,
            scheduler,
            show_sample_progress=True,
            epoch_description=f"Epoch {epoch + 1}/{args.epochs}",
            steps_per_epoch=training_dataset.steps_per_epoch(args.batch_size),
        )
        epoch_losses.append(epoch_loss)
        training_accuracies.append(training_accuracy)

        if has_validation:
            validation_correct, validation_total, validation_accuracy, validation_avg_inference_time = evaluate_model(
                spoter_model,
                validation_tf_dataset,
                args.num_classes,
            )
            validation_accuracies.append(validation_accuracy)

        metrics_postfix = {
            "loss": f"{epoch_losses[-1]:.4f}",
            "train_acc": f"{training_accuracy:.4f}",
        }

        if has_validation:
            metrics_postfix["val_acc"] = f"{validation_accuracy:.4f}"
            metrics_postfix["val_inf_ms"] = f"{validation_avg_inference_time * 1000:.2f}"

        epoch_bar.set_postfix(metrics_postfix)

        if args.save_checkpoints and training_accuracy > best_training_accuracy:
            best_training_accuracy = training_accuracy
            checkpoint_path = f"out-checkpoints/{args.experiment_name}/checkpoint_t_{checkpoint_rotation_index}.weights.h5"
            spoter_model.save_weights(checkpoint_path)

        if has_validation:
            better_validation = validation_accuracy > best_validation_accuracy
            similar_validation = math.isclose(validation_accuracy, best_validation_accuracy)
            better_loss = epoch_loss < best_validation_loss
            if better_validation or (similar_validation and better_loss):
                best_validation_accuracy = validation_accuracy
                best_validation_loss = epoch_loss
                if args.save_checkpoints:
                    spoter_model.save_weights(str(best_checkpoint_path))
                epochs_without_improvement = 0
            elif args.early_stopping_patience > 0:
                epochs_without_improvement += 1
        elif args.early_stopping_patience > 0:
            epochs_without_improvement += 1

        if epoch % args.log_freq == 0:
            tqdm.write("[" + str(epoch + 1) + "] TRAIN  loss: " + str(epoch_losses[-1]) + " acc: " + str(training_accuracy))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(epoch_losses[-1]) + " acc: " + str(training_accuracy))

            if has_validation:
                validation_message = (
                    f"[{epoch + 1}] VALIDATION  acc: {validation_accuracy:.4f}  "
                    f"avg_inf: {validation_avg_inference_time * 1000:.2f} ms"
                )
                tqdm.write(validation_message)
                logging.info(validation_message)

            tqdm.write("")
            logging.info("")

        if epoch % 10 == 0:
            best_training_accuracy, best_validation_accuracy = 0, 0
            checkpoint_rotation_index += 1

        learning_rate_history.append(float(tf.keras.backend.get_value(optimizer.learning_rate)))

        if has_validation and args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            early_stop_epoch = epoch + 1

        spoter_model.save_weights(str(latest_checkpoint_path))
        scheduler_state = scheduler.get_state() if scheduler else None
        _save_training_state(state_path, {
            "next_epoch": epoch + 1,
            "epoch_losses": epoch_losses,
            "training_accuracies": training_accuracies,
            "validation_accuracies": validation_accuracies,
            "learning_rate_history": learning_rate_history,
            "best_training_accuracy": best_training_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
            "checkpoint_rotation_index": checkpoint_rotation_index,
            "epochs_without_improvement": epochs_without_improvement,
            "scheduler_state": scheduler_state,
            "current_learning_rate": float(tf.keras.backend.get_value(optimizer.learning_rate)),
        })

        if early_stop_epoch is not None:
            break

    if early_stop_epoch is not None:
        stop_msg = f"Early stopping triggered at epoch {early_stop_epoch}"
        print(stop_msg)
        logging.info(stop_msg)

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    best_test_accuracy, best_checkpoint_name = 0, ""

    if test_tf_dataset is not None:
        for i in range(checkpoint_rotation_index):
            for checkpoint_id in ["t", "v"]:
                checkpoint_path = f"out-checkpoints/{args.experiment_name}/checkpoint_{checkpoint_id}_{i}.weights.h5"
                if not os.path.exists(checkpoint_path):
                    continue

                evaluation_model = _initialize_model(args)
                dummy_input = tf.zeros((1,) + sample_shape, dtype=tf.float32)
                evaluation_model(dummy_input, training=False)
                evaluation_model.load_weights(checkpoint_path)
                _, _, test_accuracy, test_avg_inference_time = evaluate_model(
                    evaluation_model, test_tf_dataset, args.num_classes, print_stats=True
                )

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_checkpoint_name = f"{args.experiment_name}/checkpoint_{checkpoint_id}_{i}.weights.h5"

                evaluation_message = (
                    f"checkpoint_{checkpoint_id}_{i}  ->  acc: {test_accuracy:.4f}  "
                    f"avg_inf: {test_avg_inference_time * 1000:.2f} ms"
                )
                print(evaluation_message)
                logging.info(evaluation_message)

        if best_checkpoint_path.exists():
            evaluation_model = _initialize_model(args)
            dummy_input = tf.zeros((1,) + sample_shape, dtype=tf.float32)
            evaluation_model(dummy_input, training=False)
            evaluation_model.load_weights(str(best_checkpoint_path))
            _, _, best_checkpoint_accuracy, best_checkpoint_inference = evaluate_model(
                evaluation_model, test_tf_dataset, args.num_classes, print_stats=True
            )
            if best_checkpoint_accuracy > best_test_accuracy:
                best_test_accuracy = best_checkpoint_accuracy
                best_checkpoint_name = f"{args.experiment_name}/checkpoint_best.weights.h5"
            best_checkpoint_message = (
                f"checkpoint_best  ->  acc: {best_checkpoint_accuracy:.4f}  "
                f"avg_inf: {best_checkpoint_inference * 1000:.2f} ms"
            )
            print(best_checkpoint_message)
            logging.info(best_checkpoint_message)
            export_dir = checkpoint_dir / "best_saved_model"
            if export_dir.exists():
                shutil.rmtree(export_dir)
            tf.saved_model.save(evaluation_model, export_dir)
        

        summary_message = "\nThe top result was recorded at " + str(best_test_accuracy) + " testing accuracy. The best checkpoint is " + best_checkpoint_name + "."
        print(summary_message)
        logging.info(summary_message)

    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(training_accuracies) + 1), training_accuracies, c="#00B09B", label="Training accuracy")

        if validation_dataset:
            ax.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(learning_rate_history) + 1), learning_rate_history, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[build_training_arg_parser()], add_help=False)
    args = parser.parse_args()
    run_training(args)
