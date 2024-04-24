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
    CLIPVisionLoader,
    KSampler,
    CheckpointLoaderSimple,
    VAEDecode,
    LoraLoader,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    EmptyLatentImage,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="SD1.5/epicrealism_naturalSinRC1VAE.safetensors"
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=512, height=768, batch_size=1
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_7 = cliptextencode.encode(
            text="text, watermark", clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_17 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file=None
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_19 = clipvisionloader.load_clip(clip_name=None)

        image_load = NODE_CLASS_MAPPINGS["Image Load"]()
        image_load_20 = image_load.load_image(
            image_path="./ComfyUI/input/example.png",
            RGBA="false",
            filename_text_extension="true",
        )

        loraloader = LoraLoader()
        loraloader_26 = loraloader.load_lora(
            lora_name=None,
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloader_24 = loraloader.load_lora(
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(loraloader_26, 0),
            clip=get_value_at_index(loraloader_26, 1),
        )

        loraloaderblockweight_inspire = NODE_CLASS_MAPPINGS[
            "LoraLoaderBlockWeight //Inspire"
        ]()
        loraloaderblockweight_inspire_27 = loraloaderblockweight_inspire.doit(
            category_filter="All",
            strength_model=1,
            strength_clip=1,
            inverse=False,
            seed=random.randint(1, 2**64),
            A=4,
            B=1,
            preset="Preset",
            block_vector="1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1",
            bypass=False,
            model=get_value_at_index(loraloader_24, 0),
            clip=get_value_at_index(loraloader_24, 1),
        )

        cliptextencode_25 = cliptextencode.encode(
            text="text, watermark",
            clip=get_value_at_index(loraloaderblockweight_inspire_27, 1),
        )

        ipadapterapply = NODE_CLASS_MAPPINGS["IPAdapterApply"]()
        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for q in range(10):
            ipadapterapply_18 = ipadapterapply.apply_ipadapter(
                weight=0.7000000000000001,
                noise=0.2,
                weight_type="channel penalty",
                start_at=0,
                end_at=1,
                unfold_batch=False,
                ipadapter=get_value_at_index(ipadaptermodelloader_17, 0),
                clip_vision=get_value_at_index(clipvisionloader_19, 0),
                image=get_value_at_index(image_load_20, 0),
                model=get_value_at_index(loraloaderblockweight_inspire_27, 0),
            )

            modelsamplingdiscrete_23 = modelsamplingdiscrete.patch(
                sampling="lcm",
                zsnr=False,
                model=get_value_at_index(ipadapterapply_18, 0),
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=8,
                cfg=1,
                sampler_name="lcm",
                scheduler="sgm_uniform",
                denoise=1,
                model=get_value_at_index(modelsamplingdiscrete_23, 0),
                positive=get_value_at_index(cliptextencode_25, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            image_save_16 = image_save.was_save_images(
                output_path="[time(%Y-%m-%d)]",
                filename_prefix="ComfyUI",
                filename_delimiter="_",
                filename_number_padding=2,
                filename_number_start="false",
                extension="jpg",
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
