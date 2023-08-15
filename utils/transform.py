import albumentations as A
from albumentations.pytorch import ToTensorV2

base_transform = {
    "train": A.Compose(
        [
            A.augmentations.crops.transforms.RandomCrop(224, 224, p=1.0),
            A.RandomShadow(p=0.7),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    ),
    "valid": A.Compose(
        [
            A.augmentations.crops.transforms.CenterCrop(224, 224, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ]
    ),
    "test": A.Compose([A.Normalize(), ToTensorV2()]),
}
