"""Numerically compare the local educational models with Diffusers references."""
import unittest

import torch
from diffusers import AutoencoderKL as ReferenceVAE
from diffusers import UNet2DConditionModel as ReferenceUNet
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import (
    LDMBertConfig,
    LDMBertModel as ReferenceBert,
)

from models.bert import LDMBertModel
from models.unet import ConditionalUNet
from models.vqvae import AutoencoderKL


class LocalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_bert_matches_reference(self):
        torch.manual_seed(1)
        config = dict(
            vocab_size=32,
            max_position_embeddings=8,
            d_model=16,
            encoder_layers=2,
            encoder_ffn_dim=32,
            encoder_attention_heads=2,
            head_dim=4,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            activation_function="gelu",
            pad_token_id=0,
        )
        reference = ReferenceBert(LDMBertConfig(**config)).eval()
        local = LDMBertModel(config).eval()
        local.load_state_dict(reference.state_dict(), strict=True)
        tokens = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])
        torch.testing.assert_close(local(tokens), reference(tokens)[0])

    def test_local_unet_matches_reference(self):
        torch.manual_seed(2)
        config = dict(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=16,
            attention_head_dim=4,
            norm_num_groups=32,
            norm_eps=1e-5,
            act_fn="silu",
            downsample_padding=1,
            flip_sin_to_cos=True,
            freq_shift=0,
            mid_block_scale_factor=1,
        )
        reference = ReferenceUNet(**config).eval()
        local = ConditionalUNet(config).eval()
        local.load_state_dict(reference.state_dict(), strict=True)
        latents = torch.randn(1, 4, 8, 8)
        context = torch.randn(1, 8, 16)
        expected = reference(latents, 500, encoder_hidden_states=context).sample
        actual = local(latents, 500, context)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_local_vqvae_matches_reference_encoder_and_decoder(self):
        torch.manual_seed(3)
        config = dict(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            norm_num_groups=32,
            act_fn="silu",
            scaling_factor=0.18215,
        )
        reference = ReferenceVAE(**config).eval()
        local = AutoencoderKL(config).eval()
        # New Diffusers renamed these four checkpoint paths; the published LDM
        # checkpoint and our local model retain the original explicit names.
        renamed = {}
        for key, value in reference.state_dict().items():
            key = key.replace(".to_q.", ".query.")
            key = key.replace(".to_k.", ".key.")
            key = key.replace(".to_v.", ".value.")
            key = key.replace(".to_out.0.", ".proj_attn.")
            renamed[key] = value
        local.load_state_dict(renamed, strict=True)

        image = torch.randn(1, 3, 32, 32)
        expected_moments = reference.encode(image).latent_dist.parameters
        actual_moments = local.quant_conv(local.encoder(image))
        torch.testing.assert_close(actual_moments, expected_moments, atol=2e-6, rtol=2e-6)

        latents = torch.randn(1, 4, 16, 16)
        expected_image = reference.decode(latents).sample
        actual_image = local.decode(latents)
        torch.testing.assert_close(actual_image, expected_image, atol=3e-6, rtol=3e-6)


if __name__ == "__main__":
    unittest.main()
