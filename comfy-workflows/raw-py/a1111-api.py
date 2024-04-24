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
    VAEDecode,
    NODE_CLASS_MAPPINGS,
    EmptyLatentImage,
    CheckpointLoaderSimple,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_3 = emptylatentimage.generate(
            width=1024, height=1400, batch_size=1
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
            ckpt_name="realVisionV51.safetensors"
        )

        impactwildcardencode = NODE_CLASS_MAPPINGS["ImpactWildcardEncode"]()
        impactwildcardencode_14 = impactwildcardencode.doit()

        bnk_cliptextencodeadvanced = NODE_CLASS_MAPPINGS["BNK_CLIPTextEncodeAdvanced"]()
        bnk_cliptextencodeadvanced_11 = bnk_cliptextencodeadvanced.encode(
            text="photo of a cute goth girl, hacker, sit a desk,  working at pc, skirt, clutter metro, night, neon lights, nerd outfit, cinematic, cyberpunk, clutter large room, cables",
            token_normalization="mean",
            weight_interpretation="A1111",
            clip=get_value_at_index(impactwildcardencode_14, 1),
        )

        bnk_cliptextencodeadvanced_13 = bnk_cliptextencodeadvanced.encode(
            text="makeup",
            token_normalization="mean",
            weight_interpretation="A1111",
            clip=get_value_at_index(impactwildcardencode_14, 1),
        )

        ksampler_inspire = NODE_CLASS_MAPPINGS["KSampler //Inspire"]()
        vaedecode = VAEDecode()

        for q in range(10):
            ksampler_inspire_9 = ksampler_inspire.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=3.5,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                noise_mode="GPU(=A1111)",
                model=get_value_at_index(impactwildcardencode_14, 0),
                positive=get_value_at_index(bnk_cliptextencodeadvanced_11, 0),
                negative=get_value_at_index(bnk_cliptextencodeadvanced_13, 0),
                latent_image=get_value_at_index(emptylatentimage_3, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_inspire_9, 0),
                vae=get_value_at_index(checkpointloadersimple_12, 2),
            )


if __name__ == "__main__":
    main()
