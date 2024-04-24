import os
import re
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

prrompt_template = "sexy, girl, {blonde|}, {lolita|ballgown|princess|maid}"
prompt_concat = "{pink|red|blue|green|gold|silver} glasses"
#prrompt_template = "dewiratih, {blonde|} {__color__|} {sexy|} {__dress__|__cute__|__elegant__}, {pink|} glasses"
neg="text, wAtermark"
img_folder="/content/ref1"
mask_folder="/content/ref1"
ckpt_name="RealisticVisionV51.safetensors"
lora_name="meganekko-3.ckpt"
strength_model=1.1
width=512
height=896
batch_size=2
num_images=1
num_iterations=8000 #overall repeat
cfg=1
steps=8
folder_name="dwrt-cn-[time(%Y-%m-%d-%H)]"
output_path=f"/content/drive/MyDrive/outputs/{folder_name}"
wm_folder = "/content/colab/wildcard"

strength_model_lcm=1.2
denoise=1

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
    CheckpointLoaderSimple,
    VAEDecode,
    CLIPTextEncode,
    ImageInvert,
    KSampler,
    SetLatentNoiseMask,
    NODE_CLASS_MAPPINGS,
    ConditioningConcat,
    LoraLoader,
    ImageScale,
    VAEEncode,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_2 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        loraloader = LoraLoader()
        loraloader_25 = loraloader.load_lora(
            lora_name="pytorch_lora_weights.safetensors",
            strength_model=strength_model_lcm,
            strength_clip=strength_model_lcm,
            model=get_value_at_index(checkpointloadersimple_2, 0),
            clip=get_value_at_index(checkpointloadersimple_2, 1),
        )

        loraloader_26 = loraloader.load_lora(
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_model,
            model=get_value_at_index(loraloader_25, 0),
            clip=get_value_at_index(loraloader_25, 1),
        )

        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        modelsamplingdiscrete_30 = modelsamplingdiscrete.patch(
            sampling="lcm", zsnr=False, model=get_value_at_index(loraloader_26, 0)
        )

        cliptextencode = CLIPTextEncode()
        imagescale = ImageScale()
        vaeencode = VAEEncode()
        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()
        conditioningconcat = ConditioningConcat()
        image_threshold = NODE_CLASS_MAPPINGS["Image Threshold"]()
        images_to_rgb = NODE_CLASS_MAPPINGS["Images to RGB"]()
        imageinvert = ImageInvert()
        imagecolortomask = NODE_CLASS_MAPPINGS["ImageColorToMask"]()
        setlatentnoisemask = SetLatentNoiseMask()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for q in range(num_iterations):
            wm = WildcardManager(Path(wm_folder))
            generator = RandomPromptGenerator(wildcard_manager=wm)
            prompts = list(generator.generate(prrompt_template, num_images=num_images))
            for prompt in prompts:

                cliptextencode_4 = cliptextencode.encode(
                    text=prompt_concat, clip=get_value_at_index(loraloader_26, 1)
                )

                cliptextencode_5 = cliptextencode.encode(
                    text=neg, clip=get_value_at_index(checkpointloadersimple_2, 1)
                )

                cliptextencode_83 = cliptextencode.encode(
                    text=prompt, clip=get_value_at_index(loraloader_26, 1)
                )

                conditioningconcat_84 = conditioningconcat.concat(
                    conditioning_to=get_value_at_index(cliptextencode_83, 0),
                    conditioning_from=get_value_at_index(cliptextencode_4, 0),
                )

                load_image_batch_81 = load_image_batch.load_batch_images(
                    mode="incremental_image",
                    index=0,
                    label="Batch 001",
                    path=img_folder,
                    pattern="*",
                    allow_RGBA_output="false",
                    filename_text_extension="true",
                )

                load_image_batch_82 = load_image_batch.load_batch_images(
                    mode="incremental_image",
                    index=0,
                    label="Batch 002",
                    path=mask_folder,
                    pattern="*",
                    allow_RGBA_output="false",
                    filename_text_extension="true",
                )

                imagescale_58 = imagescale.upscale(
                    upscale_method="nearest-exact",
                    width=512,
                    height=896,
                    crop="center",
                    image=get_value_at_index(load_image_batch_82, 0),
                )

                imagescale_59 = imagescale.upscale(
                    upscale_method="nearest-exact",
                    width=512,
                    height=896,
                    crop="center",
                    image=get_value_at_index(load_image_batch_81, 0),
                )

                vaeencode_86 = vaeencode.encode(
                    pixels=get_value_at_index(imagescale_59, 0),
                    vae=get_value_at_index(checkpointloadersimple_2, 2),
                )

                image_threshold_78 = image_threshold.image_threshold(
                    threshold=0.5, image=get_value_at_index(imagescale_58, 0)
                )

                images_to_rgb_77 = images_to_rgb.image_to_rgb(
                    images=get_value_at_index(image_threshold_78, 0)
                )

                imageinvert_53 = imageinvert.invert(
                    image=get_value_at_index(images_to_rgb_77, 0)
                )

                imagecolortomask_49 = imagecolortomask.image_to_mask(
                    color=0, image=get_value_at_index(imageinvert_53, 0)
                )

                setlatentnoisemask_85 = setlatentnoisemask.set_mask(
                    samples=get_value_at_index(vaeencode_86, 0),
                    mask=get_value_at_index(imagecolortomask_49, 0),
                )

                ksampler_1 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=steps,
                    cfg=cfg,
                    sampler_name="lcm",
                    scheduler="sgm_uniform",
                    denoise=denoise,
                    model=get_value_at_index(modelsamplingdiscrete_30, 0),
                    positive=get_value_at_index(conditioningconcat_84, 0),
                    negative=get_value_at_index(cliptextencode_5, 0),
                    latent_image=get_value_at_index(setlatentnoisemask_85, 0),
                )

                vaedecode_6 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_1, 0),
                    vae=get_value_at_index(checkpointloadersimple_2, 2),
                )

                image_save_29 = image_save.was_save_images(
                    output_path=output_path,
                    filename_prefix=prompt,
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
                    images=get_value_at_index(vaedecode_6, 0),
                )


if __name__ == "__main__":
    main()
