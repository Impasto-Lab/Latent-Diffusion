"""CPU tests with tiny real networks: no weights/downloads/GPU required."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, LDMTextToImagePipeline, UNet2DConditionModel
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel

import generate
from models.bert import LDMBertModel as LocalBert
from models.components import LDMComponents
from models.scheduler import DDIMScheduler as LocalDDIMScheduler
from models.unet import ConditionalUNet
from models.vqvae import AutoencoderKL as LocalAutoencoderKL


class TinyTokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        texts = [text] if isinstance(text, str) else text
        return SimpleNamespace(input_ids=torch.tensor([
            ([1] + [3 + ord(c) % 20 for c in s] + [2] + [0] * max_length)[:max_length]
            for s in texts
        ]))


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        bert_config = LDMBertConfig(vocab_size=32, max_position_embeddings=77,
                encoder_layers=1, encoder_ffn_dim=32, encoder_attention_heads=2,
                head_dim=8, d_model=16)
        bert = LDMBertModel(bert_config)
        local_bert = LocalBert(bert_config.to_dict()).eval()
        local_bert.load_state_dict(bert.state_dict(), strict=True)
        tokenizer = TinyTokenizer()
        unet_config = dict(sample_size=8, in_channels=4, out_channels=4,
                layers_per_block=1, block_out_channels=(32, 64),
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                cross_attention_dim=16, attention_head_dim=4, norm_num_groups=32,
                norm_eps=1e-5)
        unet = UNet2DConditionModel(**unet_config)
        local_unet = ConditionalUNet(unet_config).eval()
        local_unet.load_state_dict(unet.state_dict(), strict=True)
        vae_config = dict(in_channels=3, out_channels=3, latent_channels=4,
                layers_per_block=2, scaling_factor=0.18215,
                block_out_channels=(32, 32, 32, 32), norm_num_groups=32,
                down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4, sample_size=64)
        autoencoder = AutoencoderKL(**vae_config)
        local_autoencoder = LocalAutoencoderKL(vae_config).eval()
        renamed = {}
        for key, value in autoencoder.state_dict().items():
            key = key.replace(".to_q.", ".query.")
            key = key.replace(".to_k.", ".key.")
            key = key.replace(".to_v.", ".value.")
            key = key.replace(".to_out.0.", ".proj_attn.")
            renamed[key] = value
        local_autoencoder.load_state_dict(renamed, strict=True)
        scheduler = LocalDDIMScheduler(dict(
            beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000,
        ))
        cls.components = LDMComponents(
            tokenizer=tokenizer,
            text_encoder=local_bert,
            unet=local_unet,
            autoencoder=local_autoencoder,
            scheduler=scheduler,
        )
        cls.reference = LDMTextToImagePipeline(
            bert=bert,
            tokenizer=tokenizer,
            unet=unet,
            vqvae=autoencoder,
            scheduler=DDIMScheduler(
                **dict(scheduler.config), clip_sample=False, set_alpha_to_one=False,
            ),
        )
        for model in (bert, unet, autoencoder):
            model.eval()
        cls.reference.set_progress_bar_config(disable=True)

    def generate_pixels(self, **overrides):
        options = dict(height=64, width=64, steps=4, guidance=5, eta=0, seed=123)
        options.update(overrides)
        argv = ["generate.py", "--prompt", "fox", "--device", "cpu"]
        for name, value in options.items():
            argv.extend((f"--{name}", str(value)))
        with (
            patch("sys.argv", argv),
            patch("generate.load_components", return_value=self.components),
            patch("generate.save_image") as saved,
            patch("generate.tqdm", side_effect=lambda steps, **_: steps),
            patch("builtins.print"),
        ):
            generate.main()
        return saved.call_args.args[0]

    def test_matches_upstream_pipeline_with_and_without_cfg(self):
        for guidance in (1.0, 5.0):
            with self.subTest(guidance=guidance):
                actual = self.generate_pixels(guidance=guidance)
                expected = self.reference("fox", height=64, width=64, num_inference_steps=4,
                    guidance_scale=guidance, eta=0,
                    generator=torch.Generator("cpu").manual_seed(123), output_type="np").images
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_stochastic_ddim_respects_seed(self):
        a = self.generate_pixels(eta=1)
        np.testing.assert_array_equal(a, self.generate_pixels(eta=1))
        self.assertFalse(np.array_equal(a, self.generate_pixels(eta=1, seed=124)))

    def test_original_schedule_matches_author_formula(self):
        scheduler = LocalDDIMScheduler(dict(
            beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000,
        ))
        # Independently compute CompVis' betas and its final DDIM update.
        betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64)**2
        torch.testing.assert_close(scheduler.betas.double(), betas, atol=2e-9, rtol=1e-6)
        scheduler.set_timesteps(50)
        self.assertEqual(scheduler.timesteps.tolist(), list(range(1, 1000, 20))[::-1])
        alpha = scheduler.alphas_cumprod
        z, eps = torch.tensor([0.5]), torch.tensor([0.2])
        x0 = (z - (1 - alpha[1]).sqrt() * eps) / alpha[1].sqrt()
        expected = alpha[0].sqrt() * x0 + (1 - alpha[0]).sqrt() * eps
        torch.testing.assert_close(scheduler.step(eps, 1, z, eta=0), expected)

    def test_local_ddim_step_matches_diffusers_reference(self):
        config = dict(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
            timestep_spacing="leading",
        )
        local = LocalDDIMScheduler(config)
        reference = DDIMScheduler(**config)
        local.set_timesteps(50)
        reference.set_timesteps(50)
        torch.testing.assert_close(local.timesteps, reference.timesteps)

        sample = torch.randn(1, 4, 8, 8)
        predicted_noise = torch.randn_like(sample)
        timestep = local.timesteps[0]
        actual = local.step(
            predicted_noise,
            timestep,
            sample,
            eta=1,
            generator=torch.Generator("cpu").manual_seed(99),
        )
        expected = reference.step(
            predicted_noise,
            timestep,
            sample,
            eta=1,
            generator=torch.Generator("cpu").manual_seed(99),
        )
        torch.testing.assert_close(actual, expected.prev_sample)


if __name__ == "__main__":
    unittest.main()
