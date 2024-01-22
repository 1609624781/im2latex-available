import torch
import pandas as pd
from torch.utils.checkpoint import checkpoint
import numpy as np
from torch import Tensor
import torch.nn as nn
from torchvision import transforms as tvt
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pytorch_lightning as pl
from torchtext.data.metrics import bleu_score

from image2latex.model import Image2LatexModel
from data.datamodule import DataModule
from data.dataset import LatexPredictDataset
from data.text import Text100k
from data.dataset import LatexDataset

emb_dim = 80
dec_dim = 256
enc_dim = 512
attn_dim = 256
lr = 0.001
max_length = 150
log_idx = 300
max_epochs = 15
batch_size = 4
# steps_per_epoch = round(len(train_set) / batch_size)
# total_steps = steps_per_epoch * max_epochs
num_workers = 31
num_layers = 1
drop_out = 0.2
decode = "beamsearch"
beam_width=5
accumulate_batch = 64

text = Text100k()

random_state=12
torch.manual_seed(random_state)
np.random.seed(random_state)

data_path = './data/input/im2latex-sorted-by-size'
img_path = './data/input/image2latex-100k/formula_images_processed/formula_images_processed/'

train_set = LatexDataset(
    data_path=data_path,
    img_path=img_path,
    data_type="train",
    n_sample=None,
    dataset="100k",
)
val_set = LatexDataset(
    data_path=data_path,
    img_path=img_path,
    data_type="validate",
    n_sample=None,
    dataset="100k",
)
test_set = LatexDataset(
    data_path=data_path,
    img_path=img_path,
    data_type="test",
    n_sample=None,
    dataset="100k",
)

# Change predict_set to a single image (one at a time)
# predict_set = LatexPredictDataset(predict_img_path=img_path + '/6968dfca15.png')
predict_set = LatexPredictDataset(predict_img_path="./samples")
# print(predict_set)
# for image, path in predict_set:
#     print('image, path: ', image, path)

lr = 0.001
max_length = 150
log_idx = 300
max_epochs = 5
batch_size = 16
steps_per_epoch = round(len(train_set) / batch_size)
total_steps = steps_per_epoch * max_epochs
num_workers = 0

dm = DataModule(
    train_set,
    val_set,
    test_set,
    predict_set,
    num_workers,
    batch_size,
    text,
)

model = Image2LatexModel(
    total_steps=total_steps,
    lr=lr,
    n_class=text.n_class,
    enc_dim=enc_dim,
    enc_type="conv_encoder",
    emb_dim=emb_dim,
    dec_dim=dec_dim,
    attn_dim=attn_dim,
    num_layers=num_layers,
    dropout=drop_out,
    sos_id=text.sos_id,
    eos_id=text.eos_id,
    decode_type="beamsearch",
    text=text,
    beam_width=beam_width,
    log_step=100,
    log_text="store_true",
)


lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

max_epoch=15
ckpt_path = './epoch=7-step=9344.ckpt'

accumulate_grad_batches = accumulate_batch //batch_size


if __name__ == '__main__':
    trainer = pl.Trainer(
        callbacks=[lr_monitor],
        accelerator="gpu",
        devices = 1,
        log_every_n_steps=1,
        gradient_clip_val=0,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epoch,
        default_root_dir="./data/input/models/"
    )
    
    # trainer.fit(datamodule=dm, model=model, ckpt_path=ckpt_path)
    trainer.predict(datamodule=dm, model=model, ckpt_path=ckpt_path)
