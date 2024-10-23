import torch

from torch import nn
from torch.utils.data import DataLoader

from config import settings
from data import PseCoDataset
from model import PointDecoderCNN
from pretrained_models.segment_anything.build_sam import build_sam_vit_h

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = PseCoDataset(settings.TRAIN_IMAGE, settings.TRAIN_GT)
sam = build_sam_vit_h().to(device)
model = PointDecoderCNN(sam).to(device)
loss_fn = nn.MSELoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.pd.load_state_dict(torch.load(settings.MODEL_POINT_DECODER, weights_only=True))


def train(batch_size: int = 32, epochs: int = 100):
    model.train()
    data = DataLoader(dataset, batch_size=batch_size)

    for i in range(1, epochs + 1):
        train_loss = 0

        for image, points, heatmap in data:
            pred_heatmap = model(image, points)

            optimizer.zero_grad()
            loss = loss_fn(pred_heatmap, heatmap)
            loss.backward()
            optimizer.step()

            train_loss += loss / len(data)

        print(f"[Epoch {i} / {epochs}] Train loss: {train_loss}.")

        if not i % 20 or i == epochs:
            print(f"Model saved at epoch {i}.")
            torch.save(model.state_dict(), settings.MODEL_POINT_DECODER_CNN)


train()
