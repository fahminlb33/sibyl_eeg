import argparse
from termcolor import colored

from typing import List, Any

parser = argparse.ArgumentParser(description="Download and transform EEG dataset")
parser.add_argument("model",
                    type=str,
                    choices=["dense", "cnn", "lstm"],
                    help="Neural network model type")
parser.add_argument("dataset",
                    type=str,
                    help="Path to dataset file/folder")
parser.add_argument("--units",
                    type=str,
                    default="64,32",
                    help="Layer unit separated by comma (e.g. 128,64,32)")
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help="Number of epoch the neural network will be trained on")
parser.add_argument("--normalize",
                    action="store_true",
                    help="Enable batch normalization on each layer")
parser.add_argument("--dropout",
                    type=float,
                    help="Enable dropout regularization on each layer, the specified value determines dropout rate on each layer")
parser.add_argument("--save",
                    type=str,
                    help="Save latest model to the specified folder as SavedModel")
parser.add_argument("--logdir",
                    type=str,
                    help="Enable Tensorboard and save the log to the specified path")
parser.add_argument("--validation-split",
                    type=float,
                    default=0.0,
                    help="Run model validation while training, the specified value determines the validation size (0.0 - 1.0)")
parser.add_argument("--evaluation-split",
                    type=float,
                    help="Run model evaluation after training, the specified value determines the test size (0.0 - 1.0)")
parser.add_argument("--verbose",
                    action="store_true",
                    help="Enable default TensorFlow debug information")

# parse arguments
args = vars(parser.parse_args())

# main app entry point

if not args["verbose"]:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sibyl import deep_learning as dl
from sibyl.util import filesystem as fs

if not fs.is_file_exists(args["dataset"]):
    print(colored("Dataset file does not exists", "red"))
    exit()

if not fs.is_file_extension(args["dataset"], [".csv", ".parquet"]):
    print(colored("Dataset is not in .csv or .parquet extension", "red"))
    exit()

# load dataset
print(colored("Loading dataset...", "cyan"))
df: pd.DataFrame = None
if "parquet" in args["dataset"]:
    df = pd.read_parquet(args["dataset"])
else:
    df = pd.read_csv(args["dataset"])

# reshape the dataset
X, y = dl.reshape_data(df)

# prepare split data, if needed
X_train, X_test, y_train, y_test = None, None, None, None
if args["evaluation_split"] is not None:
    print(colored("\nSplitting dataset for evaluation...", "cyan"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args["evaluation_split"], stratify=y, random_state=42)

    print("Train shape: ", X_train.shape, y_train.shape)
    print("Test shape: ", X_test.shape, y_test.shape)
else:
    X_train, y_train = X, y
    print("Train shape: ", X_train.shape, y_train.shape)

# build model sequence
print(colored("\nBuilding model...", "cyan"))
model_args = {
    "kind": args["model"],
    "units": [int(x) for x in args["units"].split(",")],
    "input_shape": (X_train.shape[1], X_train.shape[2]),
    "num_classes": y_train.shape[1],
    "normalize": args["normalize"],
    "dropout": args["dropout"]
}

model = dl.build_model(**model_args)

# create tensorboard, if needed
callbacks: List[tf.keras.callbacks.Callback] = []
if args["logdir"] is not None:
    print(colored("\nPreparing TensorBoard...", "cyan"))
    callbacks.append(dl.create_tensorboard(args["logdir"]))

# compile model with an optimizer and loss function
print(colored("\nFinalizing model...", "cyan"))
dl.finalize_model(model)

print(colored("\n{} --- Model Summary ---".format((" " * 20)), "cyan"))
print(model.summary())

# start model training
print(colored("\nTraining model...", "cyan"))
dl.train_model(model, X_train, y_train, args["epochs"], args["validation_split"], callbacks)

# save model, if needed
if args["save"] is not None:
    print(colored("\nSaving model state...", "cyan"))
    tf.saved_model.save(model, args["save"])

# perform evaluation
if args["evaluation_split"] is not None:
    print(colored("\nRunning model evaluation...", "cyan"))
    loss, acc = model.evaluate(X_test, y_test)
    print("Loss: {}\nAcurracy: {}".format(loss, acc))
