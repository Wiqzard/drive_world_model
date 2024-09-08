from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import torch

# from utils.torch_utils import spatial_transform
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import nn
from torchvision import transforms as T
from vwm.modules.encoders.modules import AbstractEmbModel

from vwm.util import instantiate_from_config


def spatial_transform(
    image, z_where, out_dims, inverse=False, padding_mode="border", mode="bilinear"
):
    """
    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = (
        torch.zeros(2, 3, dtype=image.dtype)
        .repeat(image.shape[0], 1, 1)
        .to(image.device)
    )
    # set scaling
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9)

    # set translation
    theta[:, 0, -1] = (
        z_where[:, 2] if not inverse else -z_where[:, 2] / (z_where[:, 0] + 1e-9)
    )
    theta[:, 1, -1] = (
        z_where[:, 3] if not inverse else -z_where[:, 3] / (z_where[:, 1] + 1e-9)
    )
    # 2. construct sampling grid
    grid = F.affine_grid(theta, out_dims, align_corners=False)
    grid = grid.to(image.dtype)
    # 3. sample image from grid
    return F.grid_sample(
        image, grid, align_corners=False, padding_mode=padding_mode, mode=mode
    )


@dataclass
class RectangleRegion:
    x: float
    y: float
    w: float
    h: float


ImageNet_Norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def posemb_sincos_1d(n, dim, temperature=10000, dtype=torch.float32):
    """
    Creates positional embeddings for 1D patches using sin-cos positional embeddings
    Args:
        patches: 1D tensor of shape (B, N, D)
        temperature: temperature for positional embeddings
        dtype: dtype of the positional embeddings
    """

    n = torch.arange(n)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe.type(dtype)


class DINOFeaturesProvider:
    def __init__(
        self,
        dino_version: str,
        dino_channels: int,
        proj_channels: int,
        image_width: int,
        image_height: int,
        num_condition_tokens: int,
        condition_frames: List[int],
        no_condition_prob: float,
        token_dropout_prob: float,
        cage_crop: DictConfig,
        image_dropout_prob: float,
        image_token_dropout_prob: float,
        num_dino_layers: int = 1,
    ):
        super().__init__()
        self.dino_version = dino_version
        self.image_dropout_prob = image_dropout_prob
        self.image_token_dropout_prob = image_token_dropout_prob
        self.dino_channels = dino_channels
        self.num_dino_layers = num_dino_layers
        self.cage_crop = cage_crop
        self.token_dropout_prob = token_dropout_prob
        self.condition_frames = condition_frames
        self.no_condition_prob = no_condition_prob
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_version)  # .cuda()
        self.proj_channels = proj_channels

        self.patch_size = 28
        self.max_condition_tokens = len(condition_frames) * int(
            cage_crop.min_w
            * cage_crop.min_h
            * float(
                (image_width // self.patch_size) * (image_height // self.patch_size)
            )
        )
        assert (
            num_condition_tokens >= 1
            and num_condition_tokens <= self.max_condition_tokens
        ), f"num_condition_tokens should be in [1, {self.max_condition_tokens}]"

        assert (image_width // 14) % 2 == 0 and (
            image_height // 14
        ) % 2 == 0, (
            "Image width and height should be divisible by 2 after dividing by 14"
        )
        self.dino_patches_width = image_width // 28
        self.dino_patches_height = image_height // 28
        # Ensure both are divisible by 2 -> NOTE: not needed ATM since patches should be divisible by 2 already
        # self.dino_patches_width = (self.dino_patches_width - (self.dino_patches_width % 2)) // 2
        # self.dino_patches_height = (self.dino_patches_height - (self.dino_patches_height % 2)) // 2
        self.num_pos_emb_tokens = (
            self.dino_patches_width * self.dino_patches_height * len(condition_frames)
        )

        print(
            f"Using DINOv2 model with {self.num_pos_emb_tokens} positional embeddings"
        )
        self.num_condition_tokens = num_condition_tokens

    def get_features(
        self,
        target_frames: torch.Tensor,
        force_roi: RectangleRegion = None,
        force_condition_frames: torch.Tensor = None,
    ) -> torch.Tensor:
        """Given target_frames of shape [b l c h w], return the conditioning features.

        T = number of conditioning tokens over all frames
        Args:
            target_frames (torch.Tensor): [b l c h w]
        Returns:
            torch.Tensor: [b l c h w]
        """
        frames = (
            self.condition_frames
            if force_condition_frames is None
            else force_condition_frames
        )
        dino_feats = crop_and_get_dino_features(
            num_dino_layers=self.num_dino_layers,
            dino=self.dino,
            target_frames=target_frames[:, frames, ...].type(torch.float32),
            cage_crop=self.cage_crop,
            force_roi=force_roi,
        )
        return dino_feats

    def group_tokens(self, dino_feats: torch.Tensor) -> torch.Tensor:
        """Given the dino_feats, reduce the tokens by grouping them.

        Args:
            dino_feats (torch.Tensor): [b l c h w]
        Returns:
            torch.Tensor: [b l h2 w2 4 c]
        """

        # Our dino_feats have 256 channels per token, and the conditioning has 1024 channels.
        # So we group each 2x2 square of the dino_feats to get 1024 channels.
        b, l, c, h, w = dino_feats.size()
        dino_feats = dino_feats.reshape(b, l, c, h // 2, 2, w // 2, 2)
        dino_feats = dino_feats.permute(0, 1, 3, 5, 4, 6, 2).reshape(
            b, l, h // 2, w // 2, 4, c
        )
        return dino_feats

    def get_patches_to_select(self, cage_feats: torch.Tensor) -> torch.Tensor:
        """Given the cage_feats, return the patches to select.

        Args:
            cage_feats (torch.Tensor): [b l h2 w2 4 c]
        Returns:
            [torch.Tensor, torch.Tensor]: [b T], [b 2]
        """

        # Now we select self.num_condition_tokens from the VALID features
        # L = number of condition frames * number of valid patches per frame
        h = cage_feats.size(2)
        w = cage_feats.size(3)
        valid_feats = torch.any(
            torch.abs(cage_feats) > 1e-5, dim=-1
        )  # First apply torch.any over the last dimension (-1)
        valid_feats = torch.any(valid_feats, dim=-1)
        # valid_feats = torch.any(torch.abs(cage_feats) > 1e-5, dim=(-1, -2))  # [b l h2 w2]

        B = valid_feats.size(0)

        all_patches_to_select = []
        all_patches_xy = []
        for b in range(B):
            patches_to_select = torch.argwhere(valid_feats[b,].view(-1))
            patches_xy = torch.argwhere(valid_feats[b,])

            perm = torch.randperm(
                patches_to_select.size(0), device=patches_to_select.device
            )[: self.num_condition_tokens]
            patches_to_select = patches_to_select[perm]
            patches_xy = patches_xy[perm]

            all_patches_to_select.append(patches_to_select)
            all_patches_xy.append(patches_xy)

        patches_to_select = torch.stack(all_patches_to_select, dim=0).squeeze(-1)
        patches_xy = torch.stack(all_patches_xy, dim=0)
        return patches_to_select, patches_xy

    def pick_patches(
        self, dino_feats: torch.Tensor, patches_to_pick: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the DINO features and the patches to pick, return the final features for the model
        Args:
            dino_feats: [b l c h w]
            patches_to_pick
        Returns:
            torch.Tensor: [b l d]
        """
        all_dino_feats = []

        dino_feats = rearrange(dino_feats, "b l c h w -> b (l h w) c")

        for b in range(dino_feats.size(0)):
            dino_feats_b = dino_feats[b, patches_to_pick[b], :]
            all_dino_feats.append(dino_feats_b)

        dino_feats = torch.stack(all_dino_feats, dim=0)
        return dino_feats


def crop_image_for_dino_input(x: torch.Tensor):
    """Crop the image to be divisible by 14 :param x: [(bl) c h w] :return: [(bl) c h w]"""
    h, w = x.size(-2), x.size(-1)
    new_h = h - (h % 14)
    new_w = w - (w % 14)
    return x[..., :new_h, :new_w]


@torch.no_grad()
def get_dino_features(dino, x: torch.Tensor, n: int = 1):
    """Autocrops the image to be divisible by 14, and then gets the features from the DINO model.

    :param x: [(bl) c h w]
    :param n: number of layers from the last
    :return: [(bl) d h w]
    """

    x_dino = crop_image_for_dino_input(x)
    # Apply imagenet normalization mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) to each frame individually
    assert x_dino.size(1) == 3, "DINOv2 model expects 3 channels"
    x_dino = ImageNet_Norm(x_dino)
    dino_feats = dino.get_intermediate_layers(x_dino, n=n, reshape=True)
    dino_feats = torch.cat(dino_feats, dim=1)
    return dino_feats, x_dino


def create_crop_region(
    min_w: float = 0.3,
    min_h: float = 0.3,
    max_w: float = 0.8,
    max_h: float = 0.5,
    y_limit: float = 0.0,
):
    crop_w = torch.rand(1) * (max_w - min_w) + min_w
    crop_h = torch.rand(1) * (max_h - min_h) + min_h

    min_y = y_limit
    max_y = 1 - crop_h
    min_x = -1 + crop_w
    max_x = 1 - crop_w

    x = torch.rand(1) * (max_x - min_x) + min_x
    y = torch.rand(1) * (max_y - min_y) + min_y

    z_where = torch.tensor([[crop_w, crop_h, x, y]])

    return z_where


def crop_and_get_dino_features(
    num_dino_layers: int,
    dino: nn.Module,
    target_frames: torch.Tensor,
    cage_crop: DictConfig,
    force_roi: RectangleRegion = None,
) -> torch.Tensor:
    """Get the features from the DINO model and apply a random spatial transform to the target
    frames.

    Args:
        num_dino_layers (int): number of layers from the last
        dino (nn.Module): DINO model
        target_frames (torch.Tensor): [b l c h w]
    Returns:
        torch.Tensor: [b l c h w]
    """

    assert cage_crop is not None, "cage_crop should not be None"

    batch_size = target_frames.size(0)
    num_target_frames = target_frames.size(1)
    H, W = target_frames.size(-2), target_frames.size(-1)
    dtype = target_frames.dtype

    # Apply random spatial transform to target frames
    z_where = (
        create_crop_region(**cage_crop).to(dtype).to(target_frames.device)
        if force_roi is None
        else torch.tensor(
            [[force_roi.w, force_roi.h, force_roi.x, force_roi.y]],
            dtype=dtype,
            device=target_frames.device,
        )
    )
    crop_pix_w = int(W * z_where[0, 0])
    crop_pix_h = int(H * z_where[0, 1])

    flat_transformed_target_frames = spatial_transform(
        rearrange(target_frames, "b l c h w -> (b l) c h w"),
        z_where,
        [batch_size * num_target_frames, 3, crop_pix_h, crop_pix_w],
        inverse=False,
        padding_mode="border",
        mode="bilinear",
    )

    feats, cropped = get_dino_features(
        dino, flat_transformed_target_frames, n=num_dino_layers
    )

    h = H // 14
    w = W // 14

    feats_dim = feats.size(1)
    # Inverse transform features
    flat_feats = spatial_transform(
        feats,
        z_where,
        [batch_size * num_target_frames, feats_dim, h, w],
        inverse=True,
        padding_mode="zeros",
        mode="nearest",
    )
    feats = rearrange(flat_feats, "(b l) c h w -> b l c h w", b=batch_size)
    return feats


# class DinoEncoder(nn.Module):
class DinoEncoder(AbstractEmbModel):
    conditioning_provider: DINOFeaturesProvider

    def __init__(self, num_frames, condition_cfg: DictConfig):
        super().__init__()
        self.num_frames = num_frames
        self.conditioning_provider = instantiate_from_config(condition_cfg)
        # self.conditioning_provider.requires_grad_(False)
        self.cond_projection = nn.Sequential(
            nn.Linear(
                self.conditioning_provider.dino_channels,
                self.conditioning_provider.proj_channels,
            ),
            nn.LayerNorm(self.conditioning_provider.proj_channels),
        )

        self.cond_masked_tokens = nn.Parameter(
            torch.randn(self.conditioning_provider.num_condition_tokens, 1024),
            requires_grad=True,
        )
        self.cond_pos_emb = posemb_sincos_1d(
            self.conditioning_provider.num_pos_emb_tokens, 1024
        )

    def drop_pixels(
        self, pixel_values: torch.Tensor, patches_xy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_size = self.conditioning_provider.patch_size
        for coord in patches_xy:
            l, y, x = coord
            if l != 0:
                continue

            # Calculate pixel range for the patch in the grid
            x_start = x * patch_size
            x_end = (x + 1) * patch_size
            y_start = y * patch_size
            y_end = (y + 1) * patch_size

            # Set the corresponding region in the tensor to 0 for all batches and layers
            pixel_values[..., y_start:y_end, x_start:x_end] = 0
        return pixel_values

    def forward(
        self,
        pixel_values: torch.Tensor,
        # generator: torch.Generator,
        enable_pixels_dropout: bool = True,  # ignore this and handle it with self.conditioning_provider.image_token_dropout_prob
        return_grouped_tokens: bool = False,
        force_cond_feats: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConditionGenerator model that generates condition features for the given pixel values.

        Args:
            pixel_values (torch.Tensor): [ B 3 H W ]
            generator (torch.Generator): RNG generator
            enable_pixels_dropout (bool): if True, will drop patches based on condition tokens
            return_grouped_tokens (bool, optional): If set to true, returns grouped tokens for debugging/visualization. Defaults to False.
            force_roi (RectangleRegion, optional): Forces feature extraction ROI to be the given one. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cond_feats [ B N 1024 ] and pixel_values [ B 3 H W ]
        """

        if pixel_values.ndim > 4:
            raise ValueError(
                f"Expected pixel_values to have 4 dimensions, got {pixel_values.ndim}"
            )

        bs, c, h, w = pixel_values.size()
        pixel_values = pixel_values.reshape(
            bs // self.num_frames, self.num_frames, c, h, w
        )

        grouped_tokens = None
        cond_feats = (
            self.conditioning_provider.get_features(pixel_values)
            if force_cond_feats is None
            else force_cond_feats
        )
        # ensure that cond_feats' w and h is divisible by 2
        # NOTE: not needed ATM since patches should be divisible by 2 already
        # cond_feats = cond_feats[:, :, :, :cond_feats.size(3)-(cond_feats.size(3) % 2), :cond_feats.size(4)-(cond_feats.size(4) % 2)]

        b, l, c, h, w = cond_feats.size()
        cond_feats = self.conditioning_provider.group_tokens(cond_feats)
        if return_grouped_tokens:
            grouped_tokens = cond_feats.clone()
        patches_to_pick, patches_xy = self.conditioning_provider.get_patches_to_select(
            cond_feats
        )

        cond_feats = self.cond_projection(cond_feats)
        cond_feats = cond_feats.view(
            b, l, h // 2, w // 2, self.conditioning_provider.proj_channels * 4
        )
        cond_feats = rearrange(cond_feats, "b l h w c -> b (l h w) c")
        cond_feats = cond_feats + self.cond_pos_emb.to(cond_feats.device).unsqueeze(0)

        cond_feats = rearrange(
            cond_feats, "b (l h w) c -> b l c h w", l=l, h=h // 2, w=w // 2
        )
        cond_feats = self.conditioning_provider.pick_patches(
            cond_feats, patches_to_pick
        )

        if enable_pixels_dropout:
            for b in range(patches_xy.size(0)):
                sample_patches_xy = patches_xy[b]

                patches_mask = (
                    torch.rand(
                        sample_patches_xy.size(0),
                        device=pixel_values.device,
                        # generator=generator,
                    )
                    < self.conditioning_provider.image_token_dropout_prob
                )
                masked_patches_xy = sample_patches_xy[patches_mask]
                if masked_patches_xy.size(0) > 0:
                    pixel_values[b, [0], ...] = self.drop_pixels(
                        pixel_values[b, [0], ...], masked_patches_xy
                    )
        return cond_feats

        # if grouped_tokens is None:
        # return cond_feats, pixel_values
        # else:
        # return cond_feats, pixel_values, grouped_tokens
