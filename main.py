# import comet_ml at the top of your file
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import argparse
import json
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning import DDPPlugin


from transformers import RobertaModel, RobertaConfig
from transformers import RobertaForSequenceClassification

from review_cl.models import SequenceClassificationModel
from review_cl.utils import configure_model, save_transformer_model, save_pytorch_model
from review_cl.data_prep import create_data_loader


def parse_args():
    parser = argparse.ArgumentParser()
    # CLI args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--freeze_bert_layer', type=eval, default=False)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_seq_length', type=int, default=128)
    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train_data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument(
        '--validation_data', type=str, default=os.environ['SM_CHANNEL_VALIDATION']
    )
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    return parser.parse_args()


MODEL_NAME = 'model.pth'
# Hugging face list of models: https://huggingface.co/models
PRE_TRAINED_MODEL_NAME = 'roberta-base'


def main():
    # Create an experiment with your api key
    experiment = CometLogger(
        api_key="F8z2rvZxchPyTT2l1IawCAE7G",
        project_name="review-classification-bert",
        workspace="ihssen",
    )

    config = configure_model()
    roberta_model = RobertaModel.from_pretrained('roberta-base', config=config)
    model = SequenceClassificationModel(
        bert_model=roberta_model,
        num_labels=3,
        freeze_bert_layer=True,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        accelerator=None,
        num_sanity_val_steps=-1,
        val_check_interval=0.05,
        terminate_on_nan=True,
    )

    train_data_loader, df_train = create_data_loader(
        args.train_data, args.batch_size, train_or_valid="train"
    )
    val_data_loader, df_val = create_data_loader(
        args.validation_data, args.batch_size, train_or_valid="valid"
    )

    trainer.fit(model, train_data_loader, val_data_loader)

    save_transformer_model(model, args.model_dir)
    save_pytorch_model(model, args.model_dir)


if __name__ == '__main__':
    args = parse_args()
    print('Loaded arguments:')
    main()
