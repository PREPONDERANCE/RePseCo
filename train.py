import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import settings
from data import PseCoTransformedDataset
from model import PointDecoderCNN
from pretrained_models.segment_anything.build_sam import build_sam_vit_h

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = build_sam_vit_h().to(device)
# dataset = PseCoDataset(settings.TRAIN_IMAGE, settings.TRAIN_GT)
dataset = PseCoTransformedDataset(sam, settings.TRAIN_IMAGE, settings.TRAIN_GT)
model = PointDecoderCNN(sam).to(device)
loss_fn = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.pd.load_state_dict(torch.load(settings.MODEL_POINT_DECODER, weights_only=True))

del sam
torch.cuda.empty_cache()


def train(batch_size: int = 32, epochs: int = 100):
    model.train()
    data = DataLoader(dataset, batch_size=batch_size)

    for i in range(1, epochs + 1):
        train_loss = 0

        for image, points, heatmap in tqdm(data, desc=f"Epoch [{i}/{epochs}]"):
            image = image.cuda()
            heatmap = heatmap.cuda()
            points = points.cuda()

            optimizer.zero_grad()
            pred_heatmap = model(image, points)["pred_heatmaps"]
            loss = loss_fn(pred_heatmap, heatmap)
            train_loss += loss.item() / len(data)
            optimizer.step()

        print(f"[Epoch {i} / {epochs}] Train loss: {train_loss:.6f}")

        if not i % 20 or i == epochs:
            print(f"Model saved at epoch {i}.")
            torch.save(model.state_dict(), settings.MODEL_POINT_DECODER_CNN)


train(32)
