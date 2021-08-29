import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import accuracy_score


class SequenceClassificationModel(pl.LightningModule):
    def __init__(
        self,
        bert_model,
        num_labels,
        freeze_bert_layer=False,
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

        if self.freeze_bert_layer:
            print("Freezing BERT base layers...")
            for name, param in self.bert_model.named_parameters():
                if "classifier" not in name:  # classifier layer
                    param.requires_grad = False
            print("Set classifier layers to `param.requires_grad=False`.")

    def forward(self, X):
        # (B, S, H)
        hidden = self.bert_model(X)[0]
        return hidden

    def shared_step(self, batch):
        X, y = batch
        """
        if self.freeze_bert_layer:
            with torch.no_grad():
                hidden = self.bert_model(X)[0]
        hidden = hidden[:, 0]
        pooled_hidden = self.dropout(hidden)
        logits = self.classifier(pooled_hidden)
        """
        logits = self.bert_model(X)[0]
        loss = nn.CrossEntropyLoss()(logits, y)
        _, predicted = torch.max(logits.data, 1)
        # acc = ((predicted.cpu() == y.cpu()).sum()) / y.size(0)
        acc = accuracy_score(y_true=y.cpu().numpy(), y_pred=predicted.cpu().numpy())
        metrics_dict = {"loss": loss, "accuracy": acc}
        return metrics_dict

    def training_step(self, batch, batch_idx):
        metrics_dict = self.shared_step(batch)
        for k, v in metrics_dict.items():
            self.log(k, v, on_step=True, on_epoch=False, prog_bar=True)
        return metrics_dict["loss"]

    def validation_step(self, batch, batch_idx):
        metrics_dict = self.shared_step(batch)
        for k, v in metrics_dict.items():
            self.log("valid_" + k, v, on_step=False, on_epoch=True, prog_bar=True)
        return metrics_dict["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer
