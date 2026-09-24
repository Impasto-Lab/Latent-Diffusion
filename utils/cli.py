"""Command-line definition and input validation for image generation."""
import argparse
import math
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate an image with the paper authors' latent diffusion checkpoint."
    )
    parser.add_argument(
        "--prompt", default="A red fox in a snowy forest, oil painting"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="auto, cpu, mps, cuda, or a numbered GPU such as cuda:1",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--output", type=Path, help="PNG path; existing files are not overwritten"
    )
    parser.add_argument(
        "--trace-shapes",
        action="store_true",
        help="Print each major model stage's tensor shapes on its first forward",
    )
    return parser


def validate_generation_args(args):
    if not args.prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if not 1 <= args.steps < 1000:
        raise ValueError("--steps must be in [1, 999] for this checkpoint.")
    if any(size < 64 or size % 64 for size in (args.width, args.height)):
        raise ValueError(
            "Width and height must be positive multiples of 64 (recommended: 256)."
        )
    if not math.isfinite(args.guidance) or args.guidance < 0:
        raise ValueError("--guidance must be finite and nonnegative.")
    if not math.isfinite(args.eta) or not 0 <= args.eta <= 1:
        raise ValueError("--eta must be in [0, 1].")
    if not 0 <= args.seed < 2**63:
        raise ValueError("--seed must be in [0, 2**63).")
