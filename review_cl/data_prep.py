import torch
import torch.utils.datasets as tud
import pandas as pd
import numpy as np


class ReviewDataset(tud.Dataset):
    def __init__(self, input_ids_list, label_id_list):
        self.input_ids_list = input_ids_list
        self.label_id_list = label_id_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, item):
        # convert list of token_ids into an array of PyTorch LongTensors
        input_ids = json.loads(self.input_ids_list[item])
        label_id = self.label_id_list[item]

        input_ids_tensor = torch.LongTensor(input_ids)
        label_id_tensor = torch.tensor(label_id, dtype=torch.long)

        return input_ids_tensor, label_id_tensor


def create_data_loader(path, batch_size, train_or_valid="train"):
    df = pd.read_csv(path, sep="\t", usecols=["input_ids", "label_id"])
    ds = ReviewDataset(
        input_ids_list=df.input_ids.to_numpy(), label_id_list=df.label_id.to_numpy()
    )
    dataloader = tud.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train_or_valid == "train",
        drop_last=train_or_valid == "train",
    )
    return dataloader
