import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager


neg="sketch, cartoon, drawing, anime"
ckpt_name="Unstable-Diffusers-YamerMIX.safetensors"
lora_name="xl_more_art-full.safetensors"
strength_model=1
lora_weight_name="none"
lora_weight_name2="none"
strength_weight_model=0
strength_weight_model2=0
block_vector="0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0"
block_vector2="1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0"
width=1366
height=768
batch_size=2
num_images=20
rng=2
cfg=1.6
steps=6
output_path="/content/drive/MyDrive/outputs/[time(%Y-%m-%d-%H)]"
wm_folder = "/content/colab/wildcard"

def clean(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^\w_]', '', text)
    return text


prompt_lines = """__o/fruits__, __m/color__ theme surreal 3 d render, 3 d epic illustrations, 3 d artistic render,  environmental key art, surreal concept art, surrealistic digital artwork
__o/fruits__, __m/color__ theme surreal 3 d render, 3 d epic illustrations, 3 d artistic render,   key art, surreal concept art, surrealistic digital artwork
__o/*__, __m/colortheme__ theme surreal 3d render, 3d epic illustrations, 3 d artistic render,  concept key art, surreal concept art, surrealistic digital artwork
__o/landscape__, __o/landscape__ 4 k hd wallpaper very detailed, amazing wallpaper, __m/dayteme__ 4k highly detailed digital art, beautiful digital artwork, {tree|||}, no humans scenery 
__o/landscape__ 4 k hd wallpaper very detailed, amazing wallpaper, __m/dayteme__ 4k highly detailed digital art, beautiful digital artwork, {tree|||}, no humans scenery 
__o/landscape__ and __o/landscape__ 4 k hd wallpaper very detailed, amazing wallpaper, __m/dayteme__ 4k highly detailed digital art, beautiful digital artwork, {tree|||}, no humans scenery, octane render 
A sleek __m/color__ {__o/sportcar2__|__o/landtrans__|__o/trans2__|__o/landtrans__|__o/airtrans__} with a metallic finish, streaking through a __o/landscape__, __m/particle1__, at  __m/dayteme__ driving __m/lighting__
A  surrealistic {__o/fruits__|__o/bubble__|__o/vegetables__|__o/vegetables__|__o/candy__} , 2{liquid splash||} __m/colortheme__ theme
A surrealistic {floating|flying|standing|||} __o/*__ , surrounded by __m/colortheme__ theme {cloud|dust|rainbow|||}, defying gravity
A surrealistic {||__m/phase__} __o/*__ , surrounded by {__m/particle2__|||__m/glitter__}, {__m/colortheme__|} theme 
a __o/*__ , __m/colortheme__, beautiful imagery, styled product photography,  {focused macro photography||}, dramatic product photography
A surrealistic {||__m/phase__} __o/*__ , surrounded by {__m/particle2__|||__m/glitter__}, {__m/colortheme__|} theme 
a __o/*__ , __m/colortheme__, {__o/bubble__|||}, beautiful   imagery, styled product photography,  focused macro photography, dramatic product photography
A surrealistic art  of a __o/fruits__  with __m/colortheme__  {, detailed droplets||},  photograph, intricate art photography
"""
prompt_lists = prompt_lines.splitlines()


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
    NODE_CLASS_MAPPINGS,
    CheckpointLoaderSimple,
    VAEDecode,
    KSampler,
    EmptyLatentImage,
    CLIPTextEncode,
    LoraLoader,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=width, height=height, batch_size=batch_size
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_16 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        loraloader = LoraLoader()
        loraloader_18 = loraloader.load_lora(
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_model,
            model=get_value_at_index(checkpointloadersimple_16, 0),
            clip=get_value_at_index(checkpointloadersimple_16, 1),
        )

        loraloaderblockweight_inspire = NODE_CLASS_MAPPINGS[
            "LoraLoaderBlockWeight //Inspire"
        ]()
        loraloaderblockweight_inspire_32 = loraloaderblockweight_inspire.doit(
            category_filter="All",
            lora_name=lora_weight_name,
            strength_model=strength_weight_model,
            strength_clip=strength_weight_model,
            inverse=False,
            seed=random.randint(1, 2**64),
            A=1,
            B=1,
            preset="Preset",
            block_vector=block_vector,
            bypass=False,
            model=get_value_at_index(loraloader_18, 0),
            clip=get_value_at_index(loraloader_18, 1),
        )

        loraloaderblockweight_inspire_31 = loraloaderblockweight_inspire.doit(
            category_filter="All",
            lora_name=lora_weight_name2,
            strength_model=strength_weight_model2,
            strength_clip=strength_weight_model2,
            inverse=False,
            seed=random.randint(1, 2**64),
            A=1,
            B=1,
            preset="Preset",
            block_vector=block_vector2,
            bypass=False,
            model=get_value_at_index(loraloaderblockweight_inspire_32, 0),
            clip=get_value_at_index(loraloaderblockweight_inspire_32, 1),
        )

        loraloader_30 = loraloader.load_lora(
            lora_name="lcm-xl.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(loraloaderblockweight_inspire_31, 0),
            clip=get_value_at_index(loraloaderblockweight_inspire_31, 1),
        )

        cliptextencode = CLIPTextEncode()
        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for i in range(rng):
            for prrompt_template in prompt_lists:
                wm = WildcardManager(Path(wm_folder))
                generator = RandomPromptGenerator(wildcard_manager=wm)
                prompts = list(generator.generate(prrompt_template, num_images=num_images))

                for prompt in prompts:
                    cliptextencode_6 = cliptextencode.encode(
                        text=prompt,
                        clip=get_value_at_index(loraloader_30, 1),
                    )

                    cliptextencode_7 = cliptextencode.encode(
                        text=neg,
                        clip=get_value_at_index(checkpointloadersimple_16, 1),
                    )

                    modelsamplingdiscrete_27 = modelsamplingdiscrete.patch(
                        sampling="lcm", zsnr=False, model=get_value_at_index(loraloader_30, 0)
                    )

                    ksampler_29 = ksampler.sample(
                        seed=random.randint(1, 2**64),
                        steps=steps,
                        cfg=cfg,
                        sampler_name="lcm",
                        scheduler="sgm_uniform",
                        denoise=1,
                        model=get_value_at_index(modelsamplingdiscrete_27, 0),
                        positive=get_value_at_index(cliptextencode_6, 0),
                        negative=get_value_at_index(cliptextencode_7, 0),
                        latent_image=get_value_at_index(emptylatentimage_5, 0),
                    )

                    vaedecode_8 = vaedecode.decode(
                        samples=get_value_at_index(ksampler_29, 0),
                        vae=get_value_at_index(checkpointloadersimple_16, 2),
                    )

                    image_save_28 = image_save.was_save_images(
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
