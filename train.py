import os
import argparse
import random
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import __balance_val_split, __split_of_train_sequence
from datasets.czech_slr_dataset import CzechSLRDataset
from spoter.spoter_model import SPOTER, create_quantization_aware_spoter
from spoter.utils import train_epoch, evaluate, ReduceLROnPlateau
from spoter.gaussian_noise import GaussianNoise


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=64, help="Number of classes to be recognized by the model")
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
                        default="from-file", help="Type of validation set construction. See README for further reference")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=float, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=float, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    # Quantization aware training
    parser.add_argument("--quantization_aware_training", action="store_true",
                        help="Enable TensorFlow Lite quantization aware training for SPOTER")

    return parser


def _setup_logging(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _build_model(args: argparse.Namespace, sample_shape=None) -> SPOTER:
    model_kwargs = dict(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
    )

    if args.quantization_aware_training:
        if sample_shape is None:
            raise ValueError("Sample shape must be provided to initialise QAT model.")
        return create_quantization_aware_spoter(
            example_input_shape=sample_shape,
            **model_kwargs,
        )

    model = SPOTER(**model_kwargs)
    return model


def train(args):
    _set_seeds(args.seed)
    _setup_logging(args)

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"Using {device} for training.")

    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    transform = GaussianNoise(args.gaussian_mean, args.gaussian_std)
    train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=True)

    if args.validation_set == "from-file":
        val_set = CzechSLRDataset(args.validation_set_path)
    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, args.validation_set_size or 0.2)
        val_set.transform = None
        val_set.augmentations = False
    else:
        val_set = None

    if args.testing_set_path:
        eval_set = CzechSLRDataset(args.testing_set_path)
    else:
        eval_set = None

    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    sample_shape = tuple(train_set[0][0].shape)

    slrt_model = _build_model(args, sample_shape)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)

    if not args.quantization_aware_training:
        dummy_input = tf.zeros((1,) + sample_shape, dtype=tf.float32)
        slrt_model(dummy_input, training=False)

    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    for epoch in range(args.epochs):
        train_loss, _, _, train_acc = train_epoch(slrt_model, train_set, loss_fn, optimizer, scheduler)
        losses.append(train_loss / len(train_set))
        train_accs.append(train_acc)

        if val_set:
            pred_correct, pred_all, val_acc = evaluate(slrt_model, val_set)
            val_accs.append(val_acc)

        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                weights_path = f"out-checkpoints/{args.experiment_name}/checkpoint_t_{checkpoint_index}.ckpt"
                slrt_model.save_weights(weights_path)

            if val_set and val_acc > top_val_acc:
                top_val_acc = val_acc
                weights_path = f"out-checkpoints/{args.experiment_name}/checkpoint_v_{checkpoint_index}.ckpt"
                slrt_model.save_weights(weights_path)

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(losses[-1]) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(losses[-1]) + " acc: " + str(train_acc))

            if val_set:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(float(tf.keras.backend.get_value(optimizer.learning_rate)))

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_set:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                weights_path = f"out-checkpoints/{args.experiment_name}/checkpoint_{checkpoint_id}_{i}.ckpt"
                if not os.path.exists(weights_path + ".index"):
                    continue

                tested_model = _build_model(args)
                dummy_input = tf.zeros((1,) + sample_shape, dtype=tf.float32)
                tested_model(dummy_input, training=False)
                tested_model.load_weights(weights_path)
                _, _, eval_acc = evaluate(tested_model, eval_set, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + f"/checkpoint_{checkpoint_id}_{i}"

                print(f"checkpoint_{checkpoint_id}_{i}  ->  {eval_acc}")
                logging.info(f"checkpoint_{checkpoint_id}_{i}  ->  {eval_acc}")

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")

    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_set:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)
