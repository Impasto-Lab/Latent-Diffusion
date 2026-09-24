"""Typed container for the independently defined LDM components."""
from dataclasses import dataclass
from transformers import BertTokenizer

from models.bert import LDMBertModel
from models.scheduler import DDIMScheduler
from models.unet import ConditionalUNet
from models.vqvae import AutoencoderKL


@dataclass
class LDMComponents:
    """Concrete local modules that make up this LDM checkpoint."""

    tokenizer: BertTokenizer
    text_encoder: LDMBertModel
    unet: ConditionalUNet
    autoencoder: AutoencoderKL
    scheduler: DDIMScheduler

    @property
    def latent_scale_factor(self):
        return 2 ** (len(self.autoencoder.config.block_out_channels) - 1)
