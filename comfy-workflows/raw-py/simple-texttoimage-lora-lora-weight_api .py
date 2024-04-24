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
    CheckpointLoaderSimple,
    LoraLoader,
    NODE_CLASS_MAPPINGS,
    VAEDecode,
    KSampler,
    CLIPTextEncode,
    EmptyLatentImage,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="realVisionV51.safetensors"
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        loraloader = LoraLoader()
        loraloader_14 = loraloader.load_lora(
            lora_name="meganekko-3.ckpt",
            strength_model=0.8,
            strength_clip=0.8,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloaderblockweight_inspire = NODE_CLASS_MAPPINGS[
            "LoraLoaderBlockWeight //Inspire"
        ]()
        loraloaderblockweight_inspire_13 = loraloaderblockweight_inspire.doit(
            category_filter="All",
            lora_name="dwrtsg-65-512-DB2.ckpt",
            strength_model=0.6,
            strength_clip=0.6,
            inverse=False,
            seed=random.randint(1, 2**64),
            A=4,
            B=1,
            preset="Preset",
            block_vector="1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0",
            bypass=False,
            model=get_value_at_index(loraloader_14, 0),
            clip=get_value_at_index(loraloader_14, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="beautiful scenery nature glass bottle landscape, , purple galaxy bottle,",
            clip=get_value_at_index(loraloaderblockweight_inspire_13, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="text, watermark",
            clip=get_value_at_index(loraloaderblockweight_inspire_13, 1),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for q in range(10):
            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(loraloaderblockweight_inspire_13, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            image_save_10 = image_save.was_save_images(
                output_path="[time(%Y-%m-%d-%H)]",
                filename_prefix="ComfyUI",
                filename_delimiter="_",
                filename_number_padding=4,
                filename_number_start="false",
                extension="jpeg",
                quality=80,
                lossless_webp="false",
                overwrite_mode="false",
                show_history="false",
                show_history_by_prefix="true",
                embed_workflow="true",
                show_previews="true",
                images=get_value_at_index(vaedecode_8, 0),
            )


if __name__ == "__main__":
    main()
