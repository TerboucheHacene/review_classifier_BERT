import os
from transformers import RobertaModel, RobertaConfig
import torch

PRE_TRAINED_MODEL_NAME = 'roberta-base'


def configure_model():
    config = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
    config.output_attentions = True
    return config


classes_map = {-1: 0, 0: 1, 1: 2}


def create_list_input_files(path):
    input_files = glob.glob("{}/*.tsv".format(path))
    print(input_files)
    return input_files


def save_transformer_model(model, model_dir):
    path = "{}/transformer".format(model_dir)
    os.makedirs(path, exist_ok=True)
    print("Saving Transformer model to {}".format(path))
    model.save_pretrained(path)


def save_pytorch_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    print("Saving PyTorch model to {}".format(model_dir))
    save_path = os.path.join(model_dir, MODEL_NAME)
    torch.save(model.state_dict(), save_path)