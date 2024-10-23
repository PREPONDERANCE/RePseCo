import torch
import torch.nn as nn
import torchvision.ops as vision_ops

from utils.ops import _nms
from pretrained_models.segment_anything.modeling import Sam
from pretrained_models.segment_anything.modeling.mask_decoder import MLP
from pretrained_models.segment_anything.modeling.common import LayerNorm2d
from pretrained_models.segment_anything.modeling.transformer import TwoWayTransformer
from pretrained_models.segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom


class ROIHeadMLP(nn.Module):
    def __init__(self):
        super(ROIHeadMLP, self).__init__()
        self.image_region_size = 7
        self.linear = nn.Sequential(
            nn.Linear(256 * self.image_region_size * self.image_region_size, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 512),
        )

    def forward(self, features, bboxes, prompts):
        image_embeddings = vision_ops.roi_align(
            features,
            [b.reshape(-1, 4) for b in bboxes],
            output_size=(self.image_region_size, self.image_region_size),
            spatial_scale=1 / 16,
            aligned=True,
        )
        embeddings = self.linear(image_embeddings.flatten(1))
        embeddings = embeddings.reshape(-1, bboxes[0].size(1), 512)
        embeddings = torch.cat([embeddings[i].unsqueeze(0).repeat(x.size(0), 1, 1) for i, x in enumerate(prompts)])
        prompts = torch.cat(prompts)
        pred_logits = (embeddings * prompts.unsqueeze(1)).sum(dim=-1)

        return pred_logits


class PointDecoder(nn.Module):
    def __init__(
        self,
        sam: Sam,
    ) -> None:
        super().__init__()
        transformer_dim = 256

        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )
        self.mask_tokens = nn.Embedding(1, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.transformer.load_state_dict(sam.mask_decoder.transformer.state_dict())
        self.output_upscaling.load_state_dict(sam.mask_decoder.output_upscaling.state_dict())
        self.output_hypernetworks_mlp.load_state_dict(sam.mask_decoder.output_hypernetworks_mlps[0].state_dict())

        embed_dim = 256
        self.image_embedding_size = (64, 64)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.nms_kernel_size = 3
        self.point_threshold = 0.1
        self.max_points = 1000

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, image_embeddings: torch.Tensor, masks: torch.Tensor = None):
        # output_tokens: (1, 256)
        # What's the point in doing unsqueeze when you have the dimensionality?
        output_tokens = self.mask_tokens.weight[0].unsqueeze(0)

        # output_tokens: (1, 1, 256) -> (N, 1, 256) -> sparse_embeddings
        sparse_embeddings = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)

        # image_pe: (1, 256, 64, 64)
        image_pe = self.get_dense_pe()
        src = image_embeddings
        b, c, h, w = src.shape

        # hs, src (N, 4096, 256)
        # point embedding, image embedding
        hs, src = self.transformer(image_embeddings, image_pe, sparse_embeddings)

        # N, 256, 64, 64
        src = src.transpose(1, 2).view(b, c, h, w)
        # N, 256
        mask_tokens_out = hs[:, 0, :]

        # N, 32, 256, 256
        upscaled_embedding = self.output_upscaling(src)
        # N, 1, 32
        hyper_in = self.output_hypernetworks_mlp(mask_tokens_out).unsqueeze(1)
        b, c, h, w = upscaled_embedding.shape

        # N, 1, 256, 256
        pred_heatmaps = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        if self.training:
            return {"pred_heatmaps": pred_heatmaps}

        if masks is not None:
            pred_heatmaps *= masks

        with torch.no_grad():
            # pred_heatmaps_nms = _nms(pred_heatmaps.detach().sigmoid().clone(), self.nms_kernel_size)
            pred_heatmaps_nms = _nms(pred_heatmaps.detach().clone(), self.nms_kernel_size)

            # pred_points: N, 1000, 2
            # pred_points_score: N, 1000
            pred_points, pred_points_score = (
                torch.zeros(b, self.max_points, 2).cuda(),
                torch.zeros(b, self.max_points).cuda(),
            )
            m = 0
            for i in range(b):
                # (N', 2), each row indicates a non-zero position
                points = torch.nonzero((pred_heatmaps_nms[i] > self.point_threshold).squeeze())
                # (N', 2) -> (y, x)
                points = torch.flip(points, dims=(-1,))
                pred_points_score_ = pred_heatmaps_nms[i, 0, points[:, 1], points[:, 0]].flatten(0)

                idx = torch.argsort(pred_points_score_, dim=0, descending=True)[
                    : min(self.max_points, pred_points_score_.size(0))
                ]
                # print(points.size(), pred_points_score_.size(),  idx, idx.max())
                points = points[idx]
                pred_points_score_ = pred_points_score_[idx]
                # print(points.size(), pred_points_score_.size(), pred_points_score_)
                # print(pred_points.size(), pred_points_score.size())
                # print(i)
                #
                pred_points[i, : points.size(0)] = points
                pred_points_score[i, : points.size(0)] = pred_points_score_
                m = max(m, points.size(0))
            # pred_points = (pred_points + 0.5) * 4
            pred_points = pred_points[:, :m]
            pred_points_score = pred_points_score[:, :m]
            pred_points = pred_points * 4

        return {
            "pred_heatmaps": pred_heatmaps,
            "pred_points": pred_points,
            "pred_points_score": pred_points_score,
            "pred_heatmaps_nms": pred_heatmaps_nms,
        }


class PointDecoderCNN(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()

            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                LayerNorm2d(out_channels),
                nn.GELU(),
            )

        def forward(self, x: torch.Tensor):
            return self.layer(x)

    def __init__(self, sam: Sam):
        super().__init__()
        transformer_dim = 256

        self.conv = nn.Sequential(
            self.ConvBlock(transformer_dim * 2, transformer_dim * 2),
            self.ConvBlock(transformer_dim * 2, transformer_dim),
            self.ConvBlock(transformer_dim, transformer_dim),
        )
        self.pd = PointDecoder(sam)

    def forward(self, image_embeddings: torch.Tensor, points: torch.Tensor, masks: torch.Tensor = None):
        points = points // 16
        N = image_embeddings.shape[0]

        x_coords = points[..., 0].long()
        y_coords = points[..., 1].long()
        bs = torch.arange(N).unsqueeze(-1)

        # (N, 1000, 256)
        point_embedding = image_embeddings[bs, :, y_coords, x_coords]
        # (N, 1000, 1)
        point_valid = ((points[..., 0] > 0) | (points[..., 1] > 0)).unsqueeze(-1)
        # (N,)
        point_count = torch.sum(point_valid, dim=(1, 2)).unsqueeze(-1)
        # (N, 1000, 256)
        point_embedding = point_embedding * point_valid
        # (N, 256)
        point_mean = torch.sum(point_embedding, dim=1) / point_count
        # (N, 256, 1, 1)
        point_embedding = point_mean.unsqueeze(-1).unsqueeze(-1)

        ie_enhance = point_embedding * image_embeddings
        # (N, 512, 64, 64)
        image_embeddings = torch.cat([image_embeddings, ie_enhance], dim=1)

        # image embeddings: (N, 256, 64, 64)
        image_embeddings = self.conv(image_embeddings)

        return self.pd(image_embeddings, masks)
