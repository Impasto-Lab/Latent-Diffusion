"""Choose an image path and write the generated PNG."""
from datetime import datetime

from PIL import Image

from models.config import ROOT


def resolve_output_path(requested, seed):
    output = requested or ROOT / "outputs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-seed{seed}.png"
    )
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png")
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new filename.")
    return output


def save_image(pixels, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((pixels[0] * 255).round().astype("uint8"))
    image.save(output)
