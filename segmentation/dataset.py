from pytorch_lightning.core import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pycocotools import coco
from skimage import io
import os
import albumentations as A


class COCOData(LightningDataModule):
    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        stage: str = "train",
        transforms=A.Compose([A.ToFloat()]),
    ):
        super().__init__()
        self.coco = coco.COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_dir = img_dir
        self.stage = stage
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco_ = self.coco
        img_id = self.ids[index]
        ann_ids = coco_.getAnnIds(imgIds=img_id)
        target = coco_.loadAnns(ann_ids)
        mask = coco_.annToMask(target[0])
        for i in range(len(target)):
            mask += coco_.annToMask(target[i]) * i

        path = coco_.loadImgs(img_id)[0]["file_name"]

        img = io.imread(os.path.join(self.img_dir, path))
        if self.stage == "train":
            augm = self.transforms(image=img, mask=mask)
            return dict(features=augm["image"], targets=augm["mask"])
        else:
            augm = self.transforms(image=img)
            return dict(features=augm["image"])
