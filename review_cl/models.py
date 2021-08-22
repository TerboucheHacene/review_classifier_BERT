import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim


class SequenceClassificationModel(pl.LightningModule):
    def __init__(
        self,
        bert_model,
        num_labels,
        freeze_bert_layer=True,
        learning_rate=0.001,
        max_epochs=1,
        batch_size=64,
    ):
        super(SequenceClassificationModel, self).__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.embed_size = bert_model.config.hidden_size
        self.freeze_bert_layer = freeze_bert_layer
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.embed_size, num_labels)

        if self.freeze_bert_layer:
            print("Freezing BERT base layers...")
            for name, param in self.bert_model.named_parameters():
                if "classifier" not in name:  # classifier layer
                    param.requires_grad = False
            print("Set classifier layers to `param.requires_grad=False`.")

    def forward(self, X):
        # (B, S, H)
        hidden = self.bert_model(X)[0]
        # (B, H)
        hidden = hidden[:, 0]
        pooled_hidden = self.dropout(hidden)
        logits = self.classifier(pooled_hidden)
        return logits

    def shared_step(self, batch):
        X, y = batch
        if self.freeze_bert_layer:
            with torch.no_grads():
                hidden = self.bert_model(X)[0]
        hidden = hidden[:, 0]
        pooled_hidden = self.dropout(hidden)
        logits = self.classifier(pooled_hidden)
        loss = nn.CrossEntropyLoss()(logits, y)
        _, predicted = torch.max(logits.data, 1)
        acc = ((predicted.cpu() == label.cpu()).sum()) / label.size(0)
        metrics_dict = {"loss": loss, "accuracy": acc}
        return loss

    def training_step(self, batch, batch_idx):
        metrics_dict = self.shared_step(batch)
        for k, v in metrics_dict.items():
            self.log("train_" + k, v, step=True, epoch=False, sync_dist=True)
        return metrics_dict["loss"]

    def validation_step(self, X):
        metrics_dict = self.shared_step(batch)
        for k, v in metrics_dict.items():
            self.log("valid_" + k, v, step=False, epoch=True, sync_dist=True)
        return metrics_dict["loss"]

    def configure_optimizer(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer
