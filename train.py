import os
import argparse
import random
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.auto import tqdm

from utils import stratified_train_validation_split, select_training_subset
from datasets.sign_pose_dataset import SignPoseDataset
from spoter.spoter_model import SPOTER
from spoter.utils import train_single_epoch, evaluate_model, PlateauLearningRateScheduler
from spoter.gaussian_noise import AdditiveGaussianNoise


def build_training_arg_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="none", help="Type of validation set construction. See README for further reference")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used for training and evaluation dataloaders")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

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


def _set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _initialize_model(args: argparse.Namespace, sample_shape=None) -> SPOTER:
    model_kwargs = dict(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
    )

    model = SPOTER(**model_kwargs)
    return model


def run_training(args):
    _set_random_seeds(args.seed)
    _configure_logging(args)

    accelerator_type = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"Using {accelerator_type} for training.")

    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    gaussian_noise = AdditiveGaussianNoise(args.gaussian_mean, args.gaussian_std)
    training_dataset = SignPoseDataset(args.training_set_path)

    if args.validation_set == "from-file":
        validation_dataset = SignPoseDataset(args.validation_set_path)
    elif args.validation_set == "split-from-train":
        training_dataset, validation_dataset = stratified_train_validation_split(training_dataset, args.validation_set_size or 0.2)
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
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
        )
    else:
        validation_tf_dataset = None

    has_validation = validation_tf_dataset is not None

    if test_dataset:
        test_tf_dataset = test_dataset.as_tf_dataset(
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
        )
    else:
        test_tf_dataset = None

    sample_shape = training_dataset.input_shape

    spoter_model = _initialize_model(args, sample_shape)

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    scheduler = PlateauLearningRateScheduler(optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)

    dummy_input = tf.zeros((1,) + sample_shape, dtype=tf.float32)
    spoter_model(dummy_input, training=False)

    training_accuracy, validation_accuracy = 0, 0
    epoch_losses, training_accuracies, validation_accuracies = [], [], []
    learning_rate_history = []
    best_training_accuracy, best_validation_accuracy = 0, 0
    checkpoint_rotation_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", unit="epoch")
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
            validation_correct, validation_total, validation_accuracy = evaluate_model(
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

        epoch_bar.set_postfix(metrics_postfix)

        if args.save_checkpoints:
            if training_accuracy > best_training_accuracy:
                best_training_accuracy = training_accuracy
                checkpoint_path = f"out-checkpoints/{args.experiment_name}/checkpoint_t_{checkpoint_rotation_index}.weights.h5"
                spoter_model.save_weights(checkpoint_path)

            if has_validation and validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                checkpoint_path = f"out-checkpoints/{args.experiment_name}/checkpoint_v_{checkpoint_rotation_index}.weights.h5"
                spoter_model.save_weights(checkpoint_path)

        if epoch % args.log_freq == 0:
            tqdm.write("[" + str(epoch + 1) + "] TRAIN  loss: " + str(epoch_losses[-1]) + " acc: " + str(training_accuracy))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(epoch_losses[-1]) + " acc: " + str(training_accuracy))

            if has_validation:
                tqdm.write("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(validation_accuracy))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(validation_accuracy))

            tqdm.write("")
            logging.info("")

        if epoch % 10 == 0:
            best_training_accuracy, best_validation_accuracy = 0, 0
            checkpoint_rotation_index += 1

        learning_rate_history.append(float(tf.keras.backend.get_value(optimizer.learning_rate)))

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
                _, _, test_accuracy = evaluate_model(evaluation_model, test_tf_dataset, args.num_classes, print_stats=True)

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_checkpoint_name = f"{args.experiment_name}/checkpoint_{checkpoint_id}_{i}.weights.h5"

                print(f"checkpoint_{checkpoint_id}_{i}  ->  {test_accuracy}")
                logging.info(f"checkpoint_{checkpoint_id}_{i}  ->  {test_accuracy}")

        print("\nThe top result was recorded at " + str(best_test_accuracy) + " testing accuracy. The best checkpoint is " + best_checkpoint_name + ".")
        logging.info("\nThe top result was recorded at " + str(best_test_accuracy) + " testing accuracy. The best checkpoint is " + best_checkpoint_name + ".")

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
