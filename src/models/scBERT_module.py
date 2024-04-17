import numpy as np
import torch
from torch import nn
from lightning import pytorch as pl
from sklearn.metrics import accuracy_score, f1_score
from src.models.components.scBERT.performer_pytorch import PerformerLM
from torch.optim.lr_scheduler import CosineAnnealingWarmupRestarts


class Identity(torch.nn.Module):
    def __init__(self, seq_len, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=seq_len, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class scBERTCellTypeAnnotationModule(pl.LightningModule):
    """scBERT for cell type annotation. Fine-tuned with a classification head."""

    def __init__(
        self,
        learning_rate: float,
        use_pos_embed: bool,
        num_genes: int,
        num_bins: int,
    ):
        self.model = PerformerLM(
            num_tokens=num_bins + 2,
            dim=200,
            depth=6,
            max_seq_len=num_genes + 1,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=use_pos_embed,
        )
        # add classification head by overriding the output layer
        self.model.to_out = Identity(
            seq_len=num_genes+1, dropout=0., h_dim=128, out_dim=label_dict.shape[0]
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=None)

    def forward_step(self, batch, is_train=False):
        data_v, labels_v = batch
        logits = self.model(data_v)
        loss = self.loss_fn(logits, labels_v)
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        if not is_train:
            final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
            predictions = final[no_drop]
            truths = labels_v[no_drop]
            cur_acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            metrics = {'loss': loss, 'accuracy': cur_acc, 'f1': f1}
        else:
            correct_num = torch.eq(final, labels).sum(dim=-1)
            metrics = {'loss': loss, 'accuracy': correct_num / pred_num}
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.forward_step(batch, batch_idx)
        self.log(
            {
                f'train/{k}': v for k, v in metrics.items()
            }, on_step=False, on_epoch=True
        )

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.forward_step(batch, batch_idx)
        self.log(
            {
                f'val/{k}': v for k, v in metrics.items()
            }, on_step=False, on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        loss, metrics = self.forward_step(batch, batch_idx)
        self.log(
            {
                f'test/{k}': v for k, v in metrics.items()
            }, on_step=False, on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=self.learning_rate,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
