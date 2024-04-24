import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    VAEEncodeForInpaint,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    NODE_CLASS_MAPPINGS,
    KSampler,
    VAEDecode,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()
        load_image_batch_54 = load_image_batch.load_batch_images(
            mode="incremental_image",
            index=0,
            label="Batch 001",
            path="",
            pattern="*",
            allow_RGBA_output="false",
            filename_text_extension="true",
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_23 = checkpointloadersimple.load_checkpoint(
            ckpt_name="realVisionV51.safetensors"
        )

        clipseg = NODE_CLASS_MAPPINGS["CLIPSeg"]()
        clipseg_47 = clipseg.segment_image(
            text="sky",
            blur=0.1,
            threshold=0.7000000000000001,
            dilation_factor=5,
            image=get_value_at_index(load_image_batch_54, 0),
        )

        vaeencodeforinpaint = VAEEncodeForInpaint()
        vaeencodeforinpaint_21 = vaeencodeforinpaint.encode(
            grow_mask_by=0,
            pixels=get_value_at_index(load_image_batch_54, 0),
            vae=get_value_at_index(checkpointloadersimple_23, 2),
            mask=get_value_at_index(clipseg_47, 0),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_24 = cliptextencode.encode(
            text="girl, yellow hat, blue skirt, pretty, blue sunny sky",
            clip=get_value_at_index(checkpointloadersimple_23, 1),
        )

        cliptextencode_25 = cliptextencode.encode(
            text="deformed, ugly, artifacts, extra limbs, jpg",
            clip=get_value_at_index(checkpointloadersimple_23, 1),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for q in range(10):
            ksampler_22 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=7,
                sampler_name="euler_ancestral",
                scheduler="karras",
                denoise=0.8,
                model=get_value_at_index(checkpointloadersimple_23, 0),
                positive=get_value_at_index(cliptextencode_24, 0),
                negative=get_value_at_index(cliptextencode_25, 0),
                latent_image=get_value_at_index(vaeencodeforinpaint_21, 0),
            )

            vaedecode_52 = vaedecode.decode(
                samples=get_value_at_index(ksampler_22, 0),
                vae=get_value_at_index(checkpointloadersimple_23, 2),
            )

            image_save_53 = image_save.was_save_images(
                output_path="[time(%Y-%m-%d)]",
                filename_prefix="ComfyUI",
                filename_delimiter="_",
                filename_number_padding=4,
                filename_number_start="false",
                extension="jpg",
                quality=80,
                lossless_webp="false",
                overwrite_mode="false",
                show_history="false",
                show_history_by_prefix="true",
                embed_workflow="true",
                show_previews="true",
                images=get_value_at_index(vaedecode_52, 0),
            )


if __name__ == "__main__":
    main()
