import os
import torch
import numpy as np
import albumentations as A
# import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat


class PseCoDataset(Dataset):
    def __init__(self, image_dir: str, gt_dir: str, augment: bool = True):
        """
        Params
        ------
            str[image_dir]: specify the src images directory
            str[gt_dir]: specify the ground truth directory
            bool[augment]: whether to augment the image, default to `True`

        Notes
        -----
            By default, the image size will be padded to (1024, 1024) regardless
            of the value of `augment`.
        """

        self.image_dir = [
            os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir))
        ]
        self.gt_dir = [os.path.join(gt_dir, gt) for gt in sorted(os.listdir(gt_dir))]
        self.augment = augment

    def _read_image(self, image: str) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Params
        ------
            str[image]: the path of src image

        Output
        ------
            Tensor[image]: tensor representation of the image, of size (3, 1024, 1024)
            tuple[int, int]: the size of the original image

        Notes
        -----
            By default, this function will pad the src image to size (1024, 1024). If
            `augment` is set to `True`, this function will perform RandomBrightnessContrast
            at a probability of 0.3.
        """

        original_img = Image.open(image)

        augment = A.Compose(
            [
                A.LongestMaxSize(1024),
                A.PadIfNeeded(1024, 1024, position="top_left", border_mode=0, value=0),
                A.RandomBrightnessContrast(p=0.3 if self.augment else 0),
            ]
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        img = augment(image=np.array(original_img))["image"]
        # return Image.fromarray(img), original_img.size
        return transform(img), original_img.size

    def _read_gt(self, gt: str) -> torch.Tensor:
        """
        Params
        ------
            str[gt]: the ground truth label path. The ground truth are matlab annotated
            files, specify the center points of all the birds.

        Output
        ------
            Tensor[gt]: size (N, 2), each row representing the x, y coordinates.
        """

        data = loadmat(gt)
        points = data["image_info"][0][0][0][0][0]
        return torch.from_numpy(points)

    def _adjust_points(
        self, points: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Adjust the location of the points after the original
        image is padded by `albumentations`

        Params
        ------
            Tensor[points]: the annotated points of size (N, 2)
            tuple[size]: the size of the original image

        Output
        ------
            Tensor[points]: the points after adjustments
        """

        W, H = size
        scale = 1024 / W if W > H else 1024 / H
        return points * scale

    def _generate_heatmap(self, points: torch.Tensor, sigma: int = 2) -> torch.Tensor:
        """
        Generate heatmap based on the annotated points

        Params
        ------
            Tensor[points]: the annotated points after adjustments
            int[sigma]: default to 2

        Output
        ------
            Tensor[heatmap]: the generated heatmap of shape (1, 256, 256)
        """
        scale = 4
        sigma = torch.ones(points.shape[0]) * sigma
        points = (points / scale).long().float()

        x = torch.arange(0, 256, 1)
        y = torch.arange(0, 256, 1)
        x, y = torch.meshgrid(x, y, indexing="xy")
        x, y = x.unsqueeze(0), y.unsqueeze(0)

        heatmap = torch.zeros(1, 256, 256)
        for indices in torch.arange(len(points)).split(256):
            mu_x, mu_y = (
                points[indices, 0].view(-1, 1, 1),
                points[indices, 1].view(-1, 1, 1),
            )
            gaussian = torch.exp(
                -((x - mu_x) ** 2 + (y - mu_y) ** 2)
                / (2 * sigma[indices].view(-1, 1, 1) ** 2)
            )
            gaussian = torch.max(gaussian, dim=0).values
            gaussian = gaussian.reshape(1, 256, 256)
            heatmap = torch.maximum(heatmap, gaussian)

        return heatmap.float()

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index: int):
        img = self.image_dir[index]
        gt = self.gt_dir[index]

        img, size = self._read_image(img)
        points = self._adjust_points(self._read_gt(gt), size)
        htm = self._generate_heatmap(points)

        return (img, points, htm)


# img_dir = "/Users/mac/Datasets/bird_counting/train_data/images"
# gt_dir = "/Users/mac/Datasets/bird_counting/train_data/ground_truth"

# index = 219
# data = PseCoDataset(img_dir, gt_dir)
# img, gt, ht = data[index]

# plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.scatter(gt[:, 0], gt[:, 1], c="red", marker="o")
# plt.imshow(img)

# plt.subplot(1, 2, 2)
# plt.imshow(img.resize((256, 256)))
# plt.imshow(ht.squeeze().detach().numpy(), alpha=0.4)
# plt.show()
