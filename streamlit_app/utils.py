import json
import sys
import logging
import torch
from torch import nn
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
)

MODEL_NAME = "model.pth"

PRE_TRAINED_MODEL_NAME = "roberta-base"
MAX_SEQ_LEN = 128

classes = [-1, 0, 1]

TOKENIZER = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def predict_fn(review_body, model):
    model.eval()
    predicted_classes = []
    encode_plus_token = TOKENIZER.encode_plus(
        review_body,
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    input_ids = encode_plus_token["input_ids"]
    attention_mask = encode_plus_token["attention_mask"]
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        softmax_fn = nn.Softmax(dim=1)
        softmax_output = softmax_fn(output[0])

    probability_list, prediction_label_list = torch.max(softmax_output, dim=1)
    probability = probability_list.item()
    # extract the predicted label
    predicted_label_idx = prediction_label_list.item()
    predicted_label = classes[predicted_label_idx]
    # configure the response dictionary
    prediction_dict = {}
    prediction_dict["probability"] = probability
    prediction_dict["predicted_label"] = predicted_label
    prediction_dict["probabilities"] = softmax_output.numpy().tolist()[0]
    return prediction_dict
