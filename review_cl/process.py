from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import functools
import multiprocessing

from datetime import datetime
from time import gmtime, strftime, sleep

import pandas as pd
import argparse
import subprocess
import sys
import os
import re
import collections
import json
import csv
import glob
from pathlib import Path
import time

from transformers import RobertaTokenizer


# list of sentiment classes: -1 - negative; 0 - neutral; 1 - positive
classes = [-1, 0, 1]

# label IDs of the target class (sentiment) setup as a dictionary
classes_map = {-1: 0, 0: 1, 1: 2}

# tokenization model
PRE_TRAINED_MODEL_NAME = "roberta-base"

# create the tokenizer to use based on pre trained model
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def list_arg(raw_value):
    """argparse type for a list of strings"""
    return str(raw_value).split(",")


def to_sentiment(star_rating):
    if star_rating in {1, 2}:  # negative
        return -1
    if star_rating == 3:  # neutral
        return 0
    if star_rating in {4, 5}:  # positive
        return 1


# Convert the review into the BERT input ids using
def convert_to_bert_input_ids(review, max_seq_length):
    encode_plus = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_seq_length,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    return encode_plus["input_ids"].flatten().tolist()


def _preprocess_file(file, max_seq_length):
    print("file {}".format(file))
    print("max_seq_length {}".format(max_seq_length))

    # read file
    df = pd.read_csv(file, index_col=0)

    df.isna().values.any()
    df = df.dropna()
    df = df.reset_index(drop=True)
    print("Shape of dataframe {}".format(df.shape))

    # convert star rating into sentiment
    df["sentiment"] = df["Rating"].apply(
        lambda star_rating: to_sentiment(star_rating=star_rating)
    )
    print("Shape of dataframe with sentiment {}".format(df.shape))

    # convert sentiment (-1, 0, 1) into label_id (0, 1, 2)
    df["label_id"] = df["sentiment"].apply(lambda sentiment: classes_map[sentiment])
    df["input_ids"] = df["Review Text"].apply(
        lambda review: convert_to_bert_input_ids(review, max_seq_length)
    )
    # convert the index into a review_id
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "review_id", "Review Text": "review_body"})

    # drop all columns except the following:
    df = df[["review_id", "sentiment", "label_id", "input_ids", "review_body"]]
    df = df.reset_index(drop=True)
    print("Shape of dataframe after dropping columns {}".format(df.shape))

    return df


def split_df(
    df, output_data, train_split_percentage, test_split_percentage, balance_dataset
):
    # balance the dataset by sentiment down to the minority class
    if balance_dataset:

        df_unbalanced_grouped_by = df.groupby("sentiment")
        df_balanced = df_unbalanced_grouped_by.apply(
            lambda x: x.sample(df_unbalanced_grouped_by.size().min()).reset_index(
                drop=True
            )
        )
        df = df_balanced

    holdout_percentage = 1.00 - train_split_percentage
    print("holdout percentage {}".format(holdout_percentage))
    df_train, df_holdout = train_test_split(
        df, test_size=holdout_percentage, stratify=df["sentiment"]
    )

    test_holdout_percentage = test_split_percentage / holdout_percentage
    print("test holdout percentage {}".format(test_holdout_percentage))
    df_validation, df_test = train_test_split(
        df_holdout, test_size=test_holdout_percentage, stratify=df_holdout["sentiment"]
    )

    df_train = df_train.reset_index(drop=True)
    df_validation = df_validation.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_data = "{}/sentiment/train".format(output_data)
    validation_data = "{}/sentiment/validation".format(output_data)
    test_data = "{}/sentiment/test".format(output_data)

    ## write TSV Files
    df_train.to_csv("{}.tsv".format(train_data), sep="\t", index=False)
    df_validation.to_csv("{}.tsv".format(validation_data), sep="\t", index=False)
    df_test.to_csv("{}.tsv".format(test_data), sep="\t", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument(
        "--input-data", type=str, default="data/womens_clothing_ecommerce_reviews.csv"
    )
    parser.add_argument("--output-data", type=str, default="output_data")
    parser.add_argument("--train-split-percentage", type=float, default=0.90)
    parser.add_argument("--validation-split-percentage", type=float, default=0.05)
    parser.add_argument("--test-split-percentage", type=float, default=0.05)
    parser.add_argument("--balance-dataset", type=eval, default=True)
    parser.add_argument("--max-seq-length", type=int, default=128)
    return parser.parse_args()


def process(args):

    preprocessed_data = "{}/sentiment".format(args.output_data)
    df = _preprocess_file(file=args.input_data, max_seq_length=args.max_seq_length)
    split_df(
        df,
        args.output_data,
        args.train_split_percentage,
        args.test_split_percentage,
        args.balance_dataset,
    )

    print("Listing contents of {}".format(preprocessed_data))
    dirs_output = os.listdir(preprocessed_data)
    for file in dirs_output:
        print(file)

    print("Complete")


if __name__ == "__main__":

    args = parse_args()
    print("Loaded arguments:")
    process(args)
