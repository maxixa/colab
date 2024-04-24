import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager


prrompt_template = "RAW photo, {close up||} dewiratih, {blonde|} {__hairacc__|} {__hair__|}, wearing ({__color__|} {sexy|} {__dress__|__cute__|__elegant__}:1.3), {pink|} glasses, {(tutu dress:1)||} fashion photoshot, detailed,  dynamic pose, studio lghting, {white|__simbg__|__placebg__} background, (pale skin:1.3), soft lighting, film grain, Fujifilm XT3" 
neg="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
ckpt_name="RealisticVisionV51.safetensors"
lora_name="add_detail.safetensors"
strength_model=1.5
lora_weight_name="meganekko-3.ckpt"
lora_weight_name2="dwrtsg-65-512-DB2.ckpt"
strength_weight_model=0.8
strength_weight_model2=0.9
block_vector="1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0"
block_vector2="1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0"
width=512
height=896
batch_size=2
num_images = 20
cfg=6
output_path="/content/drive/MyDrive/outputs/[time(%Y-%m-%d-%H)]"
wm_folder = "/content/colab/wildcard"

def clean(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^\w_]', '', text)
    return text

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
    LoraLoader,
    CheckpointLoaderSimple,
    CLIPTextEncode,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=width, height=height, batch_size=batch_size
        )

        loraloader = LoraLoader()
        loraloader_18 = loraloader.load_lora(
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_model,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloaderblockweight_inspire = NODE_CLASS_MAPPINGS[
            "LoraLoaderBlockWeight //Inspire"
        ]()
        loraloaderblockweight_inspire_17 = loraloaderblockweight_inspire.doit(
            category_filter="All",
            lora_name=lora_weight_name,
            strength_model=strength_weight_model,
            strength_clip=1,
            inverse=False,
            seed=random.randint(1, 2**64),
            A=4,
            B=1,
            preset="Preset",
            block_vector=block_vector,
            bypass=False,
            model=get_value_at_index(loraloader_18, 0),
            clip=get_value_at_index(loraloader_18, 1),
        )

        loraloaderblockweight_inspire_16 = loraloaderblockweight_inspire.doit(
            category_filter="All",
            lora_name=lora_weight_name2,
            strength_model=strength_weight_model2,
            strength_clip=strength_weight_model2,
            inverse=False,
            seed=random.randint(1, 2**64),
            A=4,
            B=1,
            preset="Preset",
            block_vector=block_vector2,
            bypass=False,
            model=get_value_at_index(loraloaderblockweight_inspire_17, 0),
            clip=get_value_at_index(loraloaderblockweight_inspire_17, 1),
        )

        cliptextencode = CLIPTextEncode()
        ksampler_inspire = NODE_CLASS_MAPPINGS["KSampler //Inspire"]()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        
        wm = WildcardManager(Path(wm_folder))
        generator = RandomPromptGenerator(wildcard_manager=wm)
        prompts = list(generator.generate(prrompt_template, num_images=num_images))

        for prompt in prompts:
            cliptextencode_6 = cliptextencode.encode(
                text=prompt,
                clip=get_value_at_index(loraloaderblockweight_inspire_16, 1),
            )

            cliptextencode_7 = cliptextencode.encode(
                text=neg, clip=get_value_at_index(checkpointloadersimple_4, 1)
            )

            ksampler_inspire_10 = ksampler_inspire.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=cfg,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                noise_mode="GPU(=A1111)",
                model=get_value_at_index(loraloaderblockweight_inspire_16, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_inspire_10, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            image_save_11 = image_save.was_save_images(
                output_path=output_path,
                filename_prefix="dwrt",
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
                images=get_value_at_index(vaedecode_8, 0),
            )


if __name__ == "__main__":
    main()
