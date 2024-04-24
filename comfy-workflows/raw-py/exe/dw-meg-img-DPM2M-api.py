import os
import random
import sys
import re
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

prompt_template = "dewiratih, {blonde|} {__hairacc__|} {__hair__|}, {sexy|} {__dress__|__cute__|__elegant__}"

gl_color = ["pink","red","blue","","green","gold","silver",""]
neg=""
img_folder="/content/img/ref/inpaint1/2023122200"
ckpt_name="RealisticVisionV51.safetensors"
lora_name="meganekko-3ckpt"
strength_model=0
lora_weight_name="dwrtsg-65-512-DB3.ckpt"
lora_weight_name2="meganekko-3.ckpt"
strength_weight_model=1
strength_weight_model2=1
block_vector="0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0"
block_vector2="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
width=512
height=896
num_images=300 #dynamic prompt
num_iterations=1 #overall repeat
cfg=8
steps=20
denoise_range=[0.75]
output_path="/content/drive/MyDrive/outputs/img[time(%Y-%m-%d-%H)]"
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
    CLIPTextEncode,
    LoraLoader,
    ImageScale,
    VAEEncode,
    KSampler,
    VAEDecode,
    CheckpointLoaderSimple,
    NODE_CLASS_MAPPINGS,
    ConditioningConcat,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        loraloader = LoraLoader()
        loraloader_12 = loraloader.load_lora(
            lora_name="pytorch_lora_weights.safetensors",
            strength_model=0,
            strength_clip=0,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloaderblockweight_inspire = NODE_CLASS_MAPPINGS[
            "LoraLoaderBlockWeight //Inspire"
        ]()
        loraloaderblockweight_inspire_16 = loraloaderblockweight_inspire.doit(
            category_filter="All",
            lora_name=lora_weight_name,
            strength_model=strength_weight_model,
            strength_clip=strength_weight_model,
            inverse=False,
            seed=random.randint(1, 2**64),
            A=4,
            B=1,
            preset="Preset",
            block_vector=block_vector,
            bypass=False,
            model=get_value_at_index(loraloader_12, 0),
            clip=get_value_at_index(loraloader_12, 1),
        )

        loraloaderblockweight_inspire_18 = loraloaderblockweight_inspire.doit(
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
            model=get_value_at_index(loraloaderblockweight_inspire_16, 0),
            clip=get_value_at_index(loraloaderblockweight_inspire_16, 1),
        )

        cliptextencode = CLIPTextEncode()
        image_load = NODE_CLASS_MAPPINGS["Image Load"]()
        imagescale = ImageScale()
        vaeencode = VAEEncode()
        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()
        conditioningconcat = ConditioningConcat()
        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()

        for q in range(num_iterations):
            wm = WildcardManager(Path(wm_folder))
            generator = RandomPromptGenerator(wildcard_manager=wm)
            prompts = list(generator.generate(prompt_template, num_images=num_images))
            for denoise in denoise_range:
                for prompt in prompts:
                    cliptextencode_6 = cliptextencode.encode(
                        text=prompt,
                        clip=get_value_at_index(loraloaderblockweight_inspire_18, 1),
                    )
                    prompt_concat = f"{random.choice(gl_color)} glasses"
                    cliptextencode_8= cliptextencode.encode(
                        text=prompt_concat,
                        clip=get_value_at_index(loraloaderblockweight_inspire_18, 1),
                    )

                    conditioningconcat_38 = conditioningconcat.concat(
                        conditioning_to=get_value_at_index(cliptextencode_6, 0),
                        conditioning_from=get_value_at_index(cliptextencode_8, 0),
                    )

                    cliptextencode_7 = cliptextencode.encode(
                        text=neg, clip=get_value_at_index(checkpointloadersimple_4, 1)
                    )

                    load_image_batch_44 = load_image_batch.load_batch_images(
                        mode="incremental_image",
                        index=0,
                        label="Batch 001",
                        path=img_folder,
                        pattern="*",
                        allow_RGBA_output="false",
                        filename_text_extension="true",
                    )

                    imagescale_23 = imagescale.upscale(
                        upscale_method="nearest-exact",
                        width=width,
                        height=height,
                        crop="center",
                        image=get_value_at_index(load_image_batch_44, 0),
                    )

                    vaeencode_21 = vaeencode.encode(
                        pixels=get_value_at_index(imagescale_23, 0),
                        vae=get_value_at_index(checkpointloadersimple_4, 2),
                    )

                    ksampler_3 = ksampler.sample(
                        seed=random.randint(1, 2**64),
                        steps=steps,
                        cfg=cfg,
                        sampler_name="dpmpp_2m",
                        scheduler="karras",
                        denoise=denoise,
                        model=get_value_at_index(loraloaderblockweight_inspire_18, 0),
                        positive=get_value_at_index(conditioningconcat_38, 0),
                        negative=get_value_at_index(cliptextencode_7, 0),
                        latent_image=get_value_at_index(vaeencode_21, 0),
                    )

                    vaedecode_8 = vaedecode.decode(
                        samples=get_value_at_index(ksampler_3, 0),
                        vae=get_value_at_index(checkpointloadersimple_4, 2),
                    )

                    image_save_17 = image_save.was_save_images(
                        output_path=output_path,
                        filename_prefix=f"{clean(prompt)}",
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
