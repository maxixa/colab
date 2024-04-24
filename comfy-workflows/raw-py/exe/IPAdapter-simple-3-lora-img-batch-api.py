import os
import re
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

prrompt_template = " sexy, girl, glasses {blonde|}, {lolita|ballgown|mini dress}, {pink|red|blue|}"
#prrompt_template = "dewiratih, {blonde|} {__color__|} {sexy|} {__dress__|__cute__|__elegant__}, {pink|} glasses"
neg="text, wAtermark"
img_folder="/content/ref1"
img2img_folder=""
ckpt_name="RealisticVisionV51.safetensors"
lora_name="add_detail.safetensors"
strength_model=1
lora_weight_name="meganekko-3.ckpt"
strength_weight_model=0.8
block_vector="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
width=512
height=896
batch_size=2
num_images=10 #prompt gen
num_iterations=8000 #overall repeat
cfg=1
steps=8
#denoise=0.6 #img2img denoise 
denoise=random.randint(55,70)/100 #img2img denoise 
folder_out_name="dwrt-ip-[time(%Y-%m-%d-%H)]"
output_path=f"/content/drive/MyDrive/outputs/{folder_out_name}"
wm_folder = "/content/colab/wildcard"

ipadapter_file="ip-adapter-full-face_sd15.safetensors"
#ipadapter_file="ip-adapter_sd15_vit-G.bin"
#ipadapter_file="ip-adapter_sd15_light.bin"
#ipadapter_file="ip-adapter_sd15.bin"
#ipadapter_file="ip-adapter-plus_sd15.bin"
#ipadapter_file="ip-adapter-plus-face_sd15.bin"
clip_name="model.safetensors"
weight=0.55
noise=0.5
weight_type="channel penalty"

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
    CLIPVisionLoader,
    KSampler,
    CheckpointLoaderSimple,
    VAEDecode,
    LoraLoader,
    NODE_CLASS_MAPPINGS,
    RepeatLatentBatch,
    CLIPTextEncode,
    EmptyLatentImage,
    ImageScale,
    VAEEncode,
)



def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_7 = cliptextencode.encode(
            text="text, watermark", clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_17 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file=ipadapter_file
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_19 = clipvisionloader.load_clip(clip_name=clip_name)

        loraloader = LoraLoader()
        loraloader_26 = loraloader.load_lora(
            lora_name="pytorch_lora_weights.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloader_24 = loraloader.load_lora(
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_model,
            model=get_value_at_index(loraloader_26, 0),
            clip=get_value_at_index(loraloader_26, 1),
        )

        loraloaderblockweight_inspire = NODE_CLASS_MAPPINGS[
            "LoraLoaderBlockWeight //Inspire"
        ]()
        loraloaderblockweight_inspire_27 = loraloaderblockweight_inspire.doit(
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
            model=get_value_at_index(loraloader_24, 0),
            clip=get_value_at_index(loraloader_24, 1),
        )


        prepimageforclipvision = NODE_CLASS_MAPPINGS["PrepImageForClipVision"]()
        ipadapterapply = NODE_CLASS_MAPPINGS["IPAdapterApply"]()
        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        ksampler = KSampler()
        repeatlatentbatch = RepeatLatentBatch()
        vaedecode = VAEDecode()
        vaeencode = VAEEncode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()
        imagescale = ImageScale()
        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()

        for q in range(num_iterations):

            wm = WildcardManager(Path(wm_folder))
            generator = RandomPromptGenerator(wildcard_manager=wm)
            prompts = list(generator.generate(prrompt_template, num_images=num_images))
            for prompt in prompts:

                cliptextencode_25 = cliptextencode.encode(
                    text=prompt,
                    clip=get_value_at_index(loraloaderblockweight_inspire_27, 1),
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

                prepimageforclipvision_43 = prepimageforclipvision.prep_image(
                    interpolation="LANCZOS",
                    crop_position="top",
                    sharpening=0,
                    image=get_value_at_index(load_image_batch_44, 0),
                )

                load_image_batch_45 = load_image_batch.load_batch_images(
                    mode="incremental_image",
                    index=0,
                    label="Batch 002",
                    path=img2img_folder,
                    pattern="*",
                    allow_RGBA_output="false",
                    filename_text_extension="true",
                )

                imagescale_34 = imagescale.upscale(
                    upscale_method="nearest-exact",
                    width=width,
                    height=height,
                    crop="center",
                    image=get_value_at_index(load_image_batch_45, 0),
                )
                
                vaeencode_28 = vaeencode.encode(
                    pixels=get_value_at_index(imagescale_34, 0),
                    vae=get_value_at_index(checkpointloadersimple_4, 2),
                )

                repeatlatentbatch_40 = repeatlatentbatch.repeat(
                    amount=batch_size, samples=get_value_at_index(vaeencode_28, 0)
                )

                ipadapterapply_18 = ipadapterapply.apply_ipadapter(
                    weight=weight,
                    noise=noise,
                    weight_type=weight_type,
                    start_at=0,
                    end_at=1,
                    unfold_batch=False,
                    ipadapter=get_value_at_index(ipadaptermodelloader_17, 0),
                    clip_vision=get_value_at_index(clipvisionloader_19, 0),
                    image=get_value_at_index(prepimageforclipvision_43, 0),
                    model=get_value_at_index(loraloaderblockweight_inspire_27, 0),
                )

                modelsamplingdiscrete_23 = modelsamplingdiscrete.patch(
                    sampling="lcm",
                    zsnr=False,
                    model=get_value_at_index(ipadapterapply_18, 0),
                )

                ksampler_3 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=steps,
                    cfg=cfg,
                    sampler_name="lcm",
                    scheduler="sgm_uniform",
                    denoise=denoise,
                    model=get_value_at_index(modelsamplingdiscrete_23, 0),
                    positive=get_value_at_index(cliptextencode_25, 0),
                    negative=get_value_at_index(cliptextencode_7, 0),
                    latent_image=get_value_at_index(repeatlatentbatch_40, 0),
                )

                vaedecode_8 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_3, 0),
                    vae=get_value_at_index(checkpointloadersimple_4, 2),
                )

                image_save_16 = image_save.was_save_images(
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
