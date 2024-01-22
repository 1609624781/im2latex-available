from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import torchvision
from torchvision import transforms as tvt
import math
import os
import glob

class LatexDataset(Dataset):
    def __init__(
        self, data_path, img_path, data_type: str, n_sample: int = None, dataset="100k"
    ):
        """
        LaTeX数据集类

        Args:
            data_path (str): 数据集路径
            img_path (str): 图片路径
            data_type (str): 数据类型，包括"train"、"test"、"validate"
            n_sample (int, optional): 采样数量. Defaults to None.
            dataset (str, optional): 数据集名称，默认为"100k". Defaults to "100k".
        """
        super().__init__()
        assert data_type in ["train", "test", "validate"], "未找到的数据类型"
        csv_path = data_path + f"/im2latex_{data_type}.csv"
        df = pd.read_csv(csv_path)
        if n_sample:
            df = df.head(n_sample)
        df["image"] = df.image.map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        # self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])
        self.transform = tvt.Compose([tvt.Grayscale(),])

    def __len__(self):
        """
        返回数据集长度
        Returns:
            int: 数据集长度
        """
        return len(self.walker)

    def __getitem__(self, idx):
        """
        获取数据集中的一项

        Args:
            idx (int): 索引

        Returns:
            tuple: 包含图像、公式和文件名的元组
        """
        item = self.walker[idx]

        formula = item["formula"]
        image = torchvision.io.read_image(item["image"])
        # image:  torch.Size([3, 64, 256])
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)
        # 转为灰度图
        # image:  torch.Size([1, 64, 256])
        return image, formula, os.path.basename(item['image'])

class LatexPredictDataset(Dataset):
    def __init__(self, predict_img_path: str):
        """
        LaTeX预测数据集类

        Args:
            predict_img_path (str): 预测图片路径
        """
        super().__init__()
        if predict_img_path:
            assert os.path.exists(predict_img_path), "未找到图片"
            self.walker = glob.glob(predict_img_path + '/*.png')
        else:
            self.walker = glob.glob(predict_img_path + '/*.png')
        # self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])
        self.transform = tvt.Compose([tvt.Grayscale(),])

    def __len__(self):
        """
        返回预测数据集长度
        Returns:
            int: 预测数据集长度
        """
        return len(self.walker)

    def __getitem__(self, idx):
        """
        获取预测数据集中的一项

        Args:
            idx (int): 索引

        Returns:
            tuple: 包含图像和文件名的元组
        """
        img_path = self.walker[idx]

        image = torchvision.io.read_image(img_path)
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # 将图像转换为灰度图

        return image, os.path.basename(img_path)