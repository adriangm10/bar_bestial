import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2


class BarDataset(Dataset):
    def __init__(self, root_dir: str, transform: v2.Transform | None = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.masks_dir = os.path.join(root_dir, "SegmentationClass")

        self.masks = sorted([f for f in os.listdir(self.masks_dir)])
        self.images = sorted([f for f in os.listdir(self.images_dir)])

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        img_name = os.path.join(self.images_dir, self.images[idx])
        mask_name = os.path.join(self.masks_dir, self.masks[idx])

        image = decode_image(img_name).float()
        mask = decode_image(mask_name).long()

        if self.transform:
            # image, mask = self.transform(image, mask)
            seed = torch.randint(0, 10000, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)

            torch.manual_seed(seed)
            mask = self.transform(mask)
        return image, mask


if __name__ == "__main__":
    transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(),
            v2.RandomRotation((-5, 5)),
            v2.RandomGrayscale(),
        ]
    )
    dataset = BarDataset("./cv/game_images_voc/", transform=transforms)
    print(dataset[0][1].size())

    label_map = {
        0: "_background_",
        1: "q1",
        2: "q2",
        3: "q3",
        4: "q4",
        5: "q5",
        6: "h1",
        7: "h2",
        8: "h3",
        9: "h4",
        10: "hell",
        11: "heaven",
    }

    img, mask = dataset[0]
    figure = plt.figure()
    figure.add_subplot(3, 4, 1)
    plt.imshow(img.byte().permute((1, 2, 0)))
    for i in range(1, 12):
        figure.add_subplot(3, 4, i + 1)
        plt.imshow((mask == i).permute((1, 2, 0)))
        plt.title(label_map[i])

    figure = plt.figure()
    for i in range(1, 11, 2):
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, mask = dataset[idx]

        figure.add_subplot(5, 2, i)
        plt.imshow(img.byte().permute((1, 2, 0)))
        plt.axis("off")

        figure.add_subplot(5, 2, i + 1)
        plt.imshow(mask.permute((1, 2, 0)))
        plt.axis("off")

    figure.tight_layout()
    plt.show()
