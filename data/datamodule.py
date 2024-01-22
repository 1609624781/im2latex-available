import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms as tvt


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        predict_set,
        num_workers: int = 1,
        batch_size=20,
        text=None,
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.predict_set = predict_set
        self.batch_size = batch_size
        self.text = text
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

    def predict_dataloader(self):
        return DataLoader(self.predict_set, shuffle=False, batch_size=1,)

    def collate_fn(self, batch):
        # 准备整理batch的函数
        size = len(batch)
        
        # 将每个样本的文本转换为整数形式
        formulas = [self.text.text2int(i[1]) for i in batch]
        # 计算每个文本的长度，并转换为LongTensor格式
        formula_len = torch.LongTensor([i.size(-1) + 1 for i in formulas])
        # 将文本序列填充为相同长度，并设置batch_first为True
        formulas = pad_sequence(formulas, batch_first=True)
        # 添加起始符和结束符，并将数据类型转换为long
        sos = torch.zeros(size, 1) + self.text.word2id["<s>"]
        eos = torch.zeros(size, 1) + self.text.word2id["<e>"]
        formulas = torch.cat((sos, formulas, eos), dim=-1).to(dtype=torch.long)

        # 提取每个样本的图像
        images = [i[0] for i in batch]
        
        # 提取每个样本的图像名称
        images_name = [i[-1] for i in batch]
        # 计算图像宽度和高度的最大值
        max_width, max_height = 0, 0
        for img in images:
            c, h, w = img.size()
            max_width = max(max_width, w)
            max_height = max(max_height, h)

        # 定义图像填充函数
        def padding(img):
            c, h, w = img.size()
            padder = tvt.Pad((0, 0, max_width - w, max_height - h))
            return padder(img)

        # 对图像进行填充，并将数据类型转换为float
        images = torch.stack(list(map(padding, images))).to(dtype=torch.float)
        
        # # 将图像进行堆叠，并将数据类型转换为float
        # images = torch.stack(images).to(dtype=torch.float)
        
        # 返回整理后的图像、文本、文本长度和图像名称
        return images, formulas, formula_len, images_name
