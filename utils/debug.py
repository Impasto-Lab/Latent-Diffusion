"""Optional one-shot tensor-shape tracing for the local model hierarchy."""
import torch


def _tensor_shapes(value):
    if torch.is_tensor(value):
        return [str(tuple(value.shape))]
    if isinstance(value, (tuple, list)):
        shapes = []
        for item in value:
            shapes.extend(_tensor_shapes(item))
        return shapes
    return []


def install_shape_tracing(components):
    """Print each major stage once, even though the U-Net runs many times."""
    targets = [
        ("bert.token_embedding", components.text_encoder.model.embed_tokens),
        ("bert.layer.0", components.text_encoder.model.layers[0]),
        (
            f"bert.layer.{len(components.text_encoder.model.layers) - 1}",
            components.text_encoder.model.layers[-1],
        ),
        ("bert.output_norm", components.text_encoder.model.layer_norm),
        ("unet.input_conv", components.unet.conv_in),
    ]
    targets.extend(
        (f"unet.down.{index}", block)
        for index, block in enumerate(components.unet.down_blocks)
    )
    targets.append(("unet.mid", components.unet.mid_block))
    targets.extend(
        (f"unet.up.{index}", block)
        for index, block in enumerate(components.unet.up_blocks)
    )
    targets.append(("unet.output_conv", components.unet.conv_out))
    targets.extend(
        [
            ("vqvae.post_quant", components.autoencoder.post_quant_conv),
            ("vqvae.decoder.input_conv", components.autoencoder.decoder.conv_in),
            ("vqvae.decoder.mid", components.autoencoder.decoder.mid_block),
        ]
    )
    targets.extend(
        (f"vqvae.decoder.up.{index}", block)
        for index, block in enumerate(components.autoencoder.decoder.up_blocks)
    )
    targets.append(("vqvae.decoder.output_conv", components.autoencoder.decoder.conv_out))

    seen = set()
    captured_inputs = {}
    handles = []

    def make_pre_hook(label):
        def hook(module, inputs):
            if label in seen:
                return
            if label.startswith("unet.up."):
                hidden, skip_features, temb = inputs[:3]
                next_skips = reversed(skip_features[-len(module.resnets):])
                skip_shapes = ", ".join(_tensor_shapes(list(next_skips)))
                context = f", context={tuple(inputs[3].shape)}" if len(inputs) > 3 else ""
                captured_inputs[label] = (
                    f"hidden={tuple(hidden.shape)}, "
                    f"skips(pop order)=[{skip_shapes}], "
                    f"temb={tuple(temb.shape)}{context}"
                )
            else:
                captured_inputs[label] = ", ".join(_tensor_shapes(inputs)) or "-"

        return hook

    def make_post_hook(label):
        def hook(_module, _inputs, output):
            if label in seen:
                return
            seen.add(label)
            input_shapes = captured_inputs.pop(label)
            if label.startswith("unet.down."):
                hidden, new_skips = output
                skip_shapes = ", ".join(_tensor_shapes(new_skips))
                output_shapes = f"hidden={tuple(hidden.shape)}, new_skips=[{skip_shapes}]"
            else:
                output_shapes = ", ".join(_tensor_shapes(output)) or "-"
            print(f"[shape] {label}: {input_shapes} -> {output_shapes}", flush=True)

        return hook

    for label, module in targets:
        handles.append(module.register_forward_pre_hook(make_pre_hook(label)))
        handles.append(module.register_forward_hook(make_post_hook(label)))
    return handles
