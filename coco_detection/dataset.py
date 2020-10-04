from torch.utils.data import Dataset
from pycocotools import coco
from skimage import io
import os
import albumentations as A


class COCOData(Dataset):
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
        coco_ = self.coco
        img_id = self.ids[index]
        ann_ids = coco_.getAnnIds(imgIds=img_id)
        target = coco_.loadAnns(ann_ids)
        bboxes = [list(x["bbox"]) for x in target]
        labels = [x["category_id"] for x in target]
        path = coco_.loadImgs(img_id)[0]["file_name"]
        img = io.imread(os.path.join(self.img_dir, path))
        if self.stage == "train":
            augm = self.transforms(image=img, bboxes=bboxes)
            return dict(
                images=augm["image"],
                targets=dict(boxes=augm["bboxes"], labels=labels),
            )
        else:
            augm = self.transforms(image=img)
            return dict(images=augm["image"])
