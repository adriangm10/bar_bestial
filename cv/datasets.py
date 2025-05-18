import os

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode, v2


class CardDataset(Dataset):
    def __init__(self, root_dir: str, transform: v2.Transform | None = None) -> None:
        self.root_dir = root_dir
        self.transform = transform

        self.images = sorted(os.listdir(root_dir))
        self.color_labels: list[int] = []
        self.force_labels: list[int] = []
        for img in self.images:
            _, color, force, _ = img.split("_")
            self.color_labels.append(int(color))
            self.force_labels.append(int(force))

        assert len(self.color_labels) == len(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        """return: (img, (color, force))"""
        img_name = os.path.join(self.root_dir, self.images[idx])
        img = decode_image(img_name)

        if self.transform:
            img = self.transform(img)

        return img, (self.color_labels[idx], self.force_labels[idx] - 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    transforms = v2.Compose(
        [
            v2.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.2, hue=0),
            v2.RandomApply([v2.RandomRotation((180, 180))], p=0.5),
            v2.RandomRotation((-10, 10), interpolation=InterpolationMode.BILINEAR, expand=True),
            v2.RandomCrop((70, 50)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    card_dataset = CardDataset("./cv/card_images/", transform=transforms)
    color_map = {
        0: "yellow",
        1: "blue",
        2: "red",
        3: "green",
        4: "special",
    }
    figure = plt.figure()
    for i in range(1, 10):
        idx = torch.randint(len(card_dataset), size=(1,)).item()
        img, (color, force) = card_dataset[idx]

        figure.add_subplot(3, 3, i)
        plt.imshow(img.permute((1, 2, 0)))
        plt.title(color_map[color] + "_" + str(force + 1))
        plt.axis("off")

    plt.show()
