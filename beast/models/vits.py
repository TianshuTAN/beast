"""Vision transformer autoencoder implementation."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import (
    ViTMAEConfig,
    ViTMAEForPreTraining,
)
from typeguard import typechecked

from beast.models.base import BaseLightningModel
from beast.models.perceptual import AlexPerceptual
from beast import log_step


class ViTLatentMapping(nn.Module):
    """Bottleneck autoencoder for ViT patch tokens.

    Supported bottleneck_type modes:
      flat       - flatten (B, T, H) -> (B, T*H) -> Linear(T*H, L)  [legacy; ~30M params]
      cls        - take CLS token: (B, T, H) -> latent[:,0] -> Linear(H, L)
      cls_mlp    - CLS token -> 2-layer MLP Linear(H, M) -> GELU -> Dropout -> Linear(M, L);
                   layout matches the CLS-AE script so weights can be warm-started.
      pool       - mean-pool across tokens: (B, T, H) -> mean -> Linear(H, L)
      per_token  - per-token Linear(H, d_small) -> flatten -> Linear(T*d_small, L)
      patch_conv - patch tokens only (drop CLS) -> Conv2d channel reduce H->H_b per
                   position -> flatten -> Linear(P*H_b, L). Decoder mirrors and
                   prepends a learned CLS token to the reconstructed patches.
    Decoder mirrors; for cls/cls_mlp/pool, the Linear(L, H) output is broadcast
    to (B, T, H) so the ViTMAE decoder still receives a full token sequence.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        num_latents: int,
        source: str,
        bottleneck_type: str = 'flat',
        d_small: int = 8,
        mlp_hidden: int = 384,
        mlp_dropout: float = 0.0,
        patch_d_bottleneck: int | None = None,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.source = source
        self.bottleneck_type = bottleneck_type
        self.d_small = d_small
        self.mlp_hidden = mlp_hidden
        self.mlp_dropout = mlp_dropout
        self.num_patches = num_tokens - 1  # drop CLS for patch-only bottlenecks
        self.patch_d_bottleneck = (
            patch_d_bottleneck if patch_d_bottleneck is not None else max(hidden_size // 4, 64)
        )

        if source not in ('encoder', 'latents'):
            raise ValueError(f'source must be "encoder" or "latents", not {source}')

        if bottleneck_type == 'flat':
            flat = num_tokens * hidden_size
            if source == 'encoder':
                self.layer = nn.Linear(flat, num_latents)
            else:
                self.layer = nn.Linear(num_latents, flat)
        elif bottleneck_type in ('cls', 'pool'):
            if source == 'encoder':
                self.layer = nn.Linear(hidden_size, num_latents)
            else:
                self.layer = nn.Linear(num_latents, hidden_size)
        elif bottleneck_type == 'cls_mlp':
            if source == 'encoder':
                self.enc0 = nn.Linear(hidden_size, mlp_hidden)
                self.enc1 = nn.Linear(mlp_hidden, num_latents)
            else:
                self.dec0 = nn.Linear(num_latents, mlp_hidden)
                self.dec1 = nn.Linear(mlp_hidden, hidden_size)
            self.dropout = nn.Dropout(mlp_dropout)
        elif bottleneck_type == 'per_token':
            if source == 'encoder':
                self.proj = nn.Linear(hidden_size, d_small)
                self.layer = nn.Linear(num_tokens * d_small, num_latents)
            else:
                self.layer = nn.Linear(num_latents, num_tokens * d_small)
                self.unproj = nn.Linear(d_small, hidden_size)
        elif bottleneck_type == 'patch_conv':
            if source == 'encoder':
                self.reduce = nn.Conv2d(hidden_size, self.patch_d_bottleneck, kernel_size=1)
                self.layer = nn.Linear(self.num_patches * self.patch_d_bottleneck, num_latents)
            else:
                self.layer = nn.Linear(num_latents, self.num_patches * self.patch_d_bottleneck)
                self.expand = nn.Conv2d(self.patch_d_bottleneck, hidden_size, kernel_size=1)
                self.cls_placeholder = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            raise ValueError(
                f'bottleneck_type must be one of flat|cls|cls_mlp|pool|per_token|patch_conv, not {bottleneck_type}'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.source == 'encoder':
            if self.bottleneck_type == 'flat':
                return self.layer(x.reshape(x.shape[0], -1))
            if self.bottleneck_type == 'cls':
                return self.layer(x[:, 0])
            if self.bottleneck_type == 'cls_mlp':
                h = self.enc0(x[:, 0])
                h = F.gelu(h)
                h = self.dropout(h)
                return self.enc1(h)
            if self.bottleneck_type == 'pool':
                return self.layer(x.mean(dim=1))
            if self.bottleneck_type == 'per_token':
                return self.layer(self.proj(x).reshape(x.shape[0], -1))
            if self.bottleneck_type == 'patch_conv':
                # x: (B, num_tokens, H); drop CLS -> (B, num_patches, H)
                x_patch = x[:, 1:]
                # Conv2d on H as channel: (B, num_patches, H) -> (B, H, 1, num_patches)
                x_patch = x_patch.permute(0, 2, 1).unsqueeze(2)
                x_patch = self.reduce(x_patch)                   # (B, H_b, 1, num_patches)
                x_patch = x_patch.squeeze(2).permute(0, 2, 1)    # (B, num_patches, H_b)
                return self.layer(x_patch.reshape(x_patch.shape[0], -1))
        else:
            if self.bottleneck_type == 'flat':
                out = self.layer(x)
                return out.reshape(out.shape[0], self.num_tokens, self.hidden_size)
            if self.bottleneck_type in ('cls', 'pool'):
                out = self.layer(x)
                return out.unsqueeze(1).expand(-1, self.num_tokens, -1).contiguous()
            if self.bottleneck_type == 'cls_mlp':
                h = self.dec0(x)
                h = F.gelu(h)
                h = self.dropout(h)
                out = self.dec1(h)
                return out.unsqueeze(1).expand(-1, self.num_tokens, -1).contiguous()
            if self.bottleneck_type == 'per_token':
                out = self.layer(x).reshape(x.shape[0], self.num_tokens, self.d_small)
                return self.unproj(out)
            if self.bottleneck_type == 'patch_conv':
                out = self.layer(x)                                          # (B, num_patches * H_b)
                out = out.reshape(out.shape[0], self.num_patches, self.patch_d_bottleneck)
                out = out.permute(0, 2, 1).unsqueeze(2)                      # (B, H_b, 1, num_patches)
                out = self.expand(out)                                        # (B, H, 1, num_patches)
                out = out.squeeze(2).permute(0, 2, 1)                        # (B, num_patches, H)
                cls = self.cls_placeholder.expand(out.shape[0], -1, -1)      # (B, 1, H)
                return torch.cat([cls, out], dim=1).contiguous()             # (B, num_tokens, H)
        raise RuntimeError(f'unreachable: bottleneck_type={self.bottleneck_type}')


def _load_clsae_into_bottleneck(vit_mae, ckpt_path: str) -> None:
    """Warm-start a 'cls_mlp' bottleneck from a CLS-AE checkpoint.

    CLS-AE (see scripts/train_cls_ae.py::MLPAutoencoder) runs on standardized
    inputs (x - mu) / sigma. We fold (mu, sigma) into the first/last Linear's
    weight+bias so the new bottleneck operates on raw CLS features.
    """
    if vit_mae.encoder_to_latents is None or vit_mae.latents_to_decoder is None:
        raise RuntimeError('clsae_init_ckpt set but model has no bottleneck')
    if vit_mae.encoder_to_latents.bottleneck_type != 'cls_mlp':
        raise RuntimeError(
            f'clsae_init_ckpt requires bottleneck_type=cls_mlp, got '
            f'{vit_mae.encoder_to_latents.bottleneck_type}'
        )

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']
    mean = ckpt['mean'].float()               # (D_cls,)
    std = ckpt['std'].float().clamp(min=1e-3)  # (D_cls,); guard tiny sigmas

    # Encoder: MLPAutoencoder.encoder = [Linear(0), GELU(1), Dropout(2), Linear(3)]
    enc0_w = sd['encoder.0.weight'].float()   # (M, D_cls)
    enc0_b = sd['encoder.0.bias'].float()     # (M,)
    enc1_w = sd['encoder.3.weight'].float()   # (L, M)
    enc1_b = sd['encoder.3.bias'].float()     # (L,)

    # Fold input standardization into enc0:
    #   y = W0 @ ((x - mu) / sigma) + b0 = (W0 / sigma) @ x + (b0 - W0 @ (mu / sigma))
    enc0_w_folded = enc0_w / std.unsqueeze(0)
    enc0_b_folded = enc0_b - enc0_w @ (mean / std)

    # Decoder: MLPAutoencoder.decoder = [Linear(0), GELU(1), Dropout(2), Linear(3)]
    dec0_w = sd['decoder.0.weight'].float()   # (M, L)
    dec0_b = sd['decoder.0.bias'].float()     # (M,)
    dec1_w = sd['decoder.3.weight'].float()   # (D_cls, M)
    dec1_b = sd['decoder.3.bias'].float()     # (D_cls,)

    # Fold output un-standardization into dec1:
    #   x_out = sigma * (W3 @ h + b3) + mu = (sigma * W3) @ h + (sigma * b3 + mu)
    dec1_w_folded = std.unsqueeze(1) * dec1_w
    dec1_b_folded = std * dec1_b + mean

    with torch.no_grad():
        enc = vit_mae.encoder_to_latents
        dec = vit_mae.latents_to_decoder
        assert enc.enc0.weight.shape == enc0_w_folded.shape, (enc.enc0.weight.shape, enc0_w_folded.shape)
        assert enc.enc1.weight.shape == enc1_w.shape, (enc.enc1.weight.shape, enc1_w.shape)
        assert dec.dec0.weight.shape == dec0_w.shape, (dec.dec0.weight.shape, dec0_w.shape)
        assert dec.dec1.weight.shape == dec1_w_folded.shape, (dec.dec1.weight.shape, dec1_w_folded.shape)
        enc.enc0.weight.copy_(enc0_w_folded)
        enc.enc0.bias.copy_(enc0_b_folded)
        enc.enc1.weight.copy_(enc1_w)
        enc.enc1.bias.copy_(enc1_b)
        dec.dec0.weight.copy_(dec0_w)
        dec.dec0.bias.copy_(dec0_b)
        dec.dec1.weight.copy_(dec1_w_folded)
        dec.dec1.bias.copy_(dec1_b_folded)

    log_step(
        f"CLS-AE warm-start loaded from {ckpt_path}: "
        f"enc0={tuple(enc0_w_folded.shape)} enc1={tuple(enc1_w.shape)} "
        f"dec0={tuple(dec0_w.shape)} dec1={tuple(dec1_w_folded.shape)} "
        f"sigma_min={float(std.min()):.4g}",
        level='info',
    )


class BatchNormProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.embed_size)
        )

    def forward(self, x):
        proj_hidden = self.proj(x)
        return proj_hidden


@typechecked
class VisionTransformer(BaseLightningModel):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__(config)
        # Set up ViT architecture
        vit_mae_config = ViTMAEConfig(**config['model']['model_params'])

        # Check if we should use pretrained weights or random initialization
        use_pretrained = not config['model']['model_params'].get('random_init', False)
        if use_pretrained:
            log_step(
                "Loading pretrained model from 'facebook/vit-mae-base' (this may take several minutes if downloading)...", level='debug')
            log_step("Note: Model will be cached locally after first download", level='debug')
            self.vit_mae = ViTMAE.from_pretrained("facebook/vit-mae-base", config=vit_mae_config)
        else:
            log_step("Using random initialization (random_init=True)", level='debug')
            self.vit_mae = ViTMAE(vit_mae_config)
            log_step("Randomly initialized model created", level='debug')

        self.mask_ratio = config['model']['model_params']['mask_ratio']

        # Two-stage training: load BEAST checkpoint into BEAST-AE
        beast_ckpt = config['model']['model_params'].get('beast_pretrained_ckpt')
        if beast_ckpt:
            log_step(f"Loading BEAST pretrained weights from {beast_ckpt}", level='info')
            ckpt = torch.load(beast_ckpt, map_location='cpu')
            # Lightning checkpoint state_dict keys are prefixed with "vit_mae."
            beast_state = {}
            prefix = 'vit_mae.'
            for k, v in ckpt['state_dict'].items():
                if k.startswith(prefix):
                    beast_state[k[len(prefix):]] = v
            missing, unexpected = self.vit_mae.load_state_dict(beast_state, strict=False)
            log_step(
                f"BEAST weights loaded. Missing keys (expected for bottleneck): {missing}",
                level='info',
            )
            if unexpected:
                log_step(f"Unexpected keys (ignored): {unexpected}", level='warning')

        # Warm-start bottleneck from a trained CLS-AE checkpoint
        clsae_ckpt = config['model']['model_params'].get('clsae_init_ckpt')
        if clsae_ckpt:
            _load_clsae_into_bottleneck(self.vit_mae, clsae_ckpt)

        # Optionally freeze backbone (encoder + decoder), train only bottleneck
        if config['model']['model_params'].get('freeze_backbone', False):
            log_step("Freezing ViT encoder + MAE decoder (bottleneck layers remain trainable)", level='info')
            for name, param in self.vit_mae.named_parameters():
                if 'encoder_to_latents' not in name and 'latents_to_decoder' not in name:
                    param.requires_grad = False

        # perceptual loss
        if config['model']['model_params'].get('use_perceptual_loss', False):
            self.perceptual_loss = AlexPerceptual(
                device=self.device,
                criterion=nn.MSELoss()
            )
        # contrastive loss
        if config['model']['model_params'].get('use_infoNCE', False):
            self.proj = BatchNormProjector(vit_mae_config)
            if config['model']['model_params'].get('temp_scale', False):
                self.temperature = nn.Parameter(torch.ones([]) * np.log(1))

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> Dict[str, torch.Tensor]:
        results_dict = self.vit_mae(pixel_values=x, return_recon=True)
        if self.config['model']['model_params'].get('use_perceptual_loss', False):
            results_dict['perceptual_loss'] = self.perceptual_loss(
                results_dict['reconstructions'], x
            )
        if self.config['model']['model_params'].get('use_infoNCE', False):
            cls_token = results_dict['latents'][:, 0, :]
            proj_hidden = self.proj(cls_token)
            # normalize projection
            z = proj_hidden / proj_hidden.norm(dim=-1, keepdim=True)
            results_dict['z'] = z
            results_dict['cls_token'] = cls_token

        return results_dict

    def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
        x = batch_dict['image']
        results_dict = self.forward(x)
        if return_images:
            results_dict['images'] = x
        return results_dict

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss.clone()}
        ]
        loss = mse_loss
        if self.config['model']['model_params'].get('use_perceptual_loss', False):
            perceptual_loss = kwargs['perceptual_loss']
            log_list.append({
                'name': f'{stage}_perceptual',
                'value': perceptual_loss.clone()
            })
            loss += self.config['model']['model_params'].get(
                'lambda_perceptual', 10.0
            ) * perceptual_loss
        if self.config['model']['model_params'].get('use_infoNCE', False):
            z = kwargs['z']
            sim_matrix = z @ z.T
            if self.config['model']['model_params'].get('temp_scale', False):
                sim_matrix /= self.temperature.exp()
            loss_dict = batch_wise_contrastive_loss(sim_matrix)
            loss_dict['infoNCE_loss'] *= self.config['model']['model_params']['infoNCE_weight']
            log_list.append({
                'name': f'{stage}_infoNCE',
                'value': loss_dict['infoNCE_loss']
            })
            log_list.append({
                'name': f'{stage}_infoNCE_percent_correct',
                'value': loss_dict['percent_correct']
            })
            loss += loss_dict['infoNCE_loss']
        return loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        # set mask_ratio to 0 for inference
        self.vit_mae.config.mask_ratio = 0
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        # reset mask_ratio to the original value
        self.vit_mae.config.mask_ratio = self.mask_ratio
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict


class ViTMAE(ViTMAEForPreTraining):
    """ViT-MAE for masked autoencoding. Returns latents, reconstructions, and MSE loss."""

    def __init__(self, config):
        super().__init__(config)
        self.num_latents = getattr(config, 'num_latents', None)
        if self.num_latents:
            num_patches = (config.image_size // config.patch_size) ** 2
            num_visible = int(num_patches * (1 - config.mask_ratio))
            num_tokens = num_visible + 1  # +1 for CLS token
            bottleneck_type = getattr(config, 'bottleneck_type', 'flat')
            mlp_hidden = getattr(config, 'bottleneck_mlp_hidden', 384)
            mlp_dropout = getattr(config, 'bottleneck_mlp_dropout', 0.0)
            patch_d_bottleneck = getattr(config, 'bottleneck_patch_d', None)
            self.encoder_to_latents = ViTLatentMapping(
                num_tokens=num_tokens,
                hidden_size=config.hidden_size,
                num_latents=self.num_latents,
                source='encoder',
                bottleneck_type=bottleneck_type,
                mlp_hidden=mlp_hidden,
                mlp_dropout=mlp_dropout,
                patch_d_bottleneck=patch_d_bottleneck,
            )
            self.latents_to_decoder = ViTLatentMapping(
                num_tokens=num_tokens,
                hidden_size=config.hidden_size,
                num_latents=self.num_latents,
                source='latents',
                bottleneck_type=bottleneck_type,
                mlp_hidden=mlp_hidden,
                mlp_dropout=mlp_dropout,
                patch_d_bottleneck=patch_d_bottleneck,
            )
        else:
            self.encoder_to_latents = None
            self.latents_to_decoder = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_latent: bool = False,
        return_recon: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Setting default for return_dict based on the configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (self.training or self.config.mask_ratio > 0) or return_recon:
            outputs = self.vit(
                pixel_values,
                noise=noise,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            latent = outputs.last_hidden_state
        else:
            # use for fine-tuning, or inference
            # mask_ratio = 0
            embedding_output, mask, ids_restore = self.vit.embeddings(pixel_values)
            embedding_output_ = embedding_output[:, 1:, :]  # no cls token
            # unshuffle the embedding output
            index = ids_restore.unsqueeze(-1).repeat(
                1, 1, embedding_output_.shape[2]
            ).to(embedding_output_.device)
            embedding_output_ = torch.gather(embedding_output_, dim=1, index=index)
            # add cls token back
            embedding_output = torch.cat((embedding_output[:, :1, :], embedding_output_), dim=1)
            encoder_outputs = self.vit.encoder(
                embedding_output,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            latent = self.vit.layernorm(sequence_output)
            if not return_latent:
                # return the cls token and 0 loss if not return_latent
                return latent[:, 0], 0
        if return_latent:
            return latent
        # extract cls latent
        cls_latent = latent[:, 0]  # shape (batch_size, hidden_size)
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        # BEAST-AE bottleneck: compress patch tokens to low-dim latents, then expand back
        bottleneck_latents = None
        if self.encoder_to_latents is not None:
            bottleneck_latents = self.encoder_to_latents(latent)   # (B, num_latents)
            latent = self.latents_to_decoder(bottleneck_latents)   # (B, T, H)

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits
        # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        if self.config.mask_ratio == 0:
            # parent forward_loss does (loss*mask).sum()/mask.sum() = 0/0 = NaN
            # when nothing is masked. For pure-AE mode, average over all patches.
            target = self.patchify(pixel_values)
            if self.config.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5
            loss = ((logits - target) ** 2).mean()
        else:
            loss = self.forward_loss(pixel_values, logits, mask)

        if return_recon:
            result = {
                'latents': latent,
                'loss': loss,
                'mse_loss': loss,
                'reconstructions': self.unpatchify(logits),
            }
            if bottleneck_latents is not None:
                result['bottleneck_latents'] = bottleneck_latents
            return result
        result = {
            'latents': cls_latent,
            'loss': loss,
            'logits': logits,
        }
        if bottleneck_latents is not None:
            result['bottleneck_latents'] = bottleneck_latents
        return result


def topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, axis=1)[:, -(i+1)] == labels) / len(labels)
    return topsum


def batch_wise_contrastive_loss(sim_matrix):
    N = sim_matrix.shape[0]
    # remove the diagonal from the sim_matrix
    mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix = sim_matrix[~mask].view(N, N-1)
    labels = torch.arange(N).to(sim_matrix.device)
    labels_i, labels_j = labels[:N//2], labels[N//2:] - 1
    labels = torch.cat([labels_j, labels_i]).to(sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    percent_correct = topk(sim_matrix, labels, k=1)
    return {
        "infoNCE_loss": loss,
        "percent_correct": percent_correct,
    }
