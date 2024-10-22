from torch.utils.data import DataLoader

from model import PointDecoder
from data import PseCoDataset

img_dir = "data/train_data/images"
gt_dir = "data/train_data/ground_truth"

dataset = PseCoDataset(img_dir, gt_dir)
data = DataLoader(dataset, batch_size=4)
model = PointDecoder()
