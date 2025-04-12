import os
import time

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode, v2


class BarDataset(Dataset):
    def __init__(self, root_dir: str, transform: v2.Transform | None = None, combine_qs_and_hs: bool = False) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.combine_qs_and_hs = combine_qs_and_hs
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.masks_dir = os.path.join(root_dir, "SegmentationClass")

        self.masks = sorted([f for f in os.listdir(self.masks_dir)])
        self.images = sorted([f for f in os.listdir(self.images_dir)])

        assert len(self.images) == len(self.masks)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        mask_name = os.path.join(self.masks_dir, self.masks[idx])

        image = decode_image(img_name)
        mask = decode_image(mask_name).long()

        if self.transform:
            seed = time.time_ns()
            torch.manual_seed(seed)
            image = self.transform(image)

            torch.manual_seed(seed)
            mask = self.transform(mask)

        if self.combine_qs_and_hs:
            mask[(1 <= mask) & (mask <= 5)] = 1
            mask[(6 <= mask) & (mask <= 9)] = 2
            mask[10 == mask] = 3
            mask[11 == mask] = 4

        return image, mask


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
            v2.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation((-5, 5)),
            v2.RandomGrayscale(),
        ]
    )
    combine_qs_and_hs = True
    dataset = BarDataset("./cv/game_images_voc/", transform=transforms, combine_qs_and_hs=combine_qs_and_hs)
    print(dataset[0][1].size())

    if combine_qs_and_hs:
        label_map = {
            0: "_background_",
            1: "q",
            2: "h",
            3: "hell",
            4: "heaven",
        }
    else:
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
    args = (2, 3) if combine_qs_and_hs else (4, 3)
    figure = plt.figure()
    figure.add_subplot(*args, 1)
    plt.imshow(img.permute((1, 2, 0)))
    for i in range(1, len(label_map)):
        figure.add_subplot(*args, i + 1)
        plt.imshow((mask == i).permute((1, 2, 0)))
        plt.title(label_map[i])

    figure = plt.figure()
    for i in range(1, 10):
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, mask = dataset[idx]

        figure.add_subplot(3, 3, i)
        plt.imshow(img.permute((1, 2, 0)))
        plt.imshow(mask.permute((1, 2, 0)), alpha=0.7)
        plt.axis("off")

    figure.tight_layout()
    plt.show()

    transforms = v2.Compose(
        [
            v2.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.2, hue=0),
            v2.RandomVerticalFlip(),
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
