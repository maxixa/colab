import os
import re
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

prrompt_template = "{close up|}, sexy, girl, {blonde|}, {lolita|ballgown|princess|maid}, {pink|red|blue|} glasses"
ckpt_name="RealisticVisionV51.safetensors"
width=512
height=896
batch_size=2
num_images=1 #num prompt generated
num_iterations=8000 #overall repeat

control_net_name="control_v11p_sd15_inpaint_fp16.safetensors"
controlnet_strength=1
lora_name="meganekko-3.ckpt"
strength_model=1
img_folder=""
mask_folder=""
steps=8
cfg=1
output_path="/content/drive/MyDrive/outputs/dwrt-inpaint-[time(%Y-%m-%d-%H)]"
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
    CheckpointLoaderSimple,
    CLIPTextEncode,
    KSampler,
    LoraLoader,
    EmptyLatentImage,
    ImageInvert,
    VAEDecode,
    ControlNetApply,
    NODE_CLASS_MAPPINGS,
    ControlNetLoader,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=width, height=height, batch_size=batch_size
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_33 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_7 = cliptextencode.encode(
            text=" text, error, cropped, (worst quality:1.2), (low quality:1.2), normal quality, ",
            clip=get_value_at_index(checkpointloadersimple_33, 1),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_31 = controlnetloader.load_controlnet(
            control_net_name=control_net_name
        )

        loraloader = LoraLoader()
        loraloader_133 = loraloader.load_lora(
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_model,
            model=get_value_at_index(checkpointloadersimple_33, 0),
            clip=get_value_at_index(checkpointloadersimple_33, 1),
        )

        loraloader_132 = loraloader.load_lora(
            lora_name="pytorch_lora_weights.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(loraloader_133, 0),
            clip=get_value_at_index(loraloader_133, 1),
        )



        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        imageinvert = ImageInvert()
        imagecolortomask = NODE_CLASS_MAPPINGS["ImageColorToMask"]()
        inpaintpreprocessor = NODE_CLASS_MAPPINGS["InpaintPreprocessor"]()
        controlnetapply = ControlNetApply()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for q in range(num_iterations):
            wm = WildcardManager(Path(wm_folder))
            generator = RandomPromptGenerator(wildcard_manager=wm)
            prompts = list(generator.generate(prrompt_template, num_images=num_images))
            for prompt in prompts:
                modelsamplingdiscrete_130 = modelsamplingdiscrete.patch(
                    sampling="lcm", zsnr=False, model=get_value_at_index(loraloader_132, 0)
                )

                cliptextencode_131 = cliptextencode.encode(
                    text=prompt, clip=get_value_at_index(loraloader_132, 1)
                )

                load_image_batch_134 = load_image_batch.load_batch_images(
                    mode="incremental_image",
                    index=0,
                    label="Batch 001",
                    path=img_folder,
                    pattern="*",
                    allow_RGBA_output="false",
                    filename_text_extension="true",
                )

                load_image_batch_135 = load_image_batch.load_batch_images(
                    mode="incremental_image",
                    index=0,
                    label="Batch 002",
                    path=mask_folder,
                    pattern="*",
                    allow_RGBA_output="false",
                    filename_text_extension="true",
                )

                # imageinvert_128 = imageinvert.invert(
                #     image=get_value_at_index(load_image_batch_135, 0)
                # )

                imagecolortomask_129 = imagecolortomask.image_to_mask(
                    color=0, image=get_value_at_index(load_image_batch_135, 0)
                )

                inpaintpreprocessor_99 = inpaintpreprocessor.preprocess(
                    image=get_value_at_index(load_image_batch_134, 0),
                    mask=get_value_at_index(imagecolortomask_129, 0),
                )

                controlnetapply_23 = controlnetapply.apply_controlnet(
                    strength=controlnet_strength,
                    conditioning=get_value_at_index(cliptextencode_131, 0),
                    control_net=get_value_at_index(controlnetloader_31, 0),
                    image=get_value_at_index(inpaintpreprocessor_99, 0),
                )

                ksampler_3 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=steps,
                    cfg=cfg,
                    sampler_name="lcm",
                    scheduler="sgm_uniform",
                    denoise=1,
                    model=get_value_at_index(modelsamplingdiscrete_130, 0),
                    positive=get_value_at_index(controlnetapply_23, 0),
                    negative=get_value_at_index(cliptextencode_7, 0),
                    latent_image=get_value_at_index(emptylatentimage_5, 0),
                )

                vaedecode_8 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_3, 0),
                    vae=get_value_at_index(checkpointloadersimple_33, 2),
                )

                image_save_36 = image_save.was_save_images(
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
