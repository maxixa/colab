import os
import random
import re
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager

g_template = "{|gold|silver} (glasses:1)"
# g_template = "{red|pink|white|gold|silver} (glasses:1)"
prrompt_template = "cinematic, full body, glamour photo of woman, {||twintails}, {blonde|}, {bow hair|cat ears|Bows|Hair Clips|Headbands|Hair Ties|Barrettes|Hair Slides|Ponytail Holders|Hair Pins|Flower Crowns|Bobby Pins|Hair Sticks|Hair Combs|Scrunchies|Hair Tassels|Crown Headbands|Hair Charms|Braided Headbands|Hair Wraps|Ponytail Streamers|Glitter Hair Ties}, (sexy:1.3){|pink|red|peach|maroon|light-blue|Navy|Scarlet|Royal-blue|Turquoise|Olive|Emerald|Sage|Gold|Cream|Purple|Lavender|Violet|Brown|Tan|Blush|Rose|Fuchsia|Magenta|pink||} {lolita dress|fairy dress, wings|princess dress|ballgown|wedding dress| BUTTERFLY DRESS, wings|BURLESQUE DRESS|CUTE mini DRESS|FLOWER DRESS|SAILOR SENSHI UNIFORM| VICTORIAN DRESS| VICTORIAN mini DRESS} ,{||thighhighs||stockings|}, {Sashes|Ruffles|Bows|Ribbons|Lace Trims|Petticoats|Tutus|Belts|Buckles|Brooches|Flower Pins|Appliques|Embroidery|Patches|Ribbon Bows|Dress Clips|Waist Belts|Dress Pins|Dress Brooches|Dress Sashes}, {tutu|(tutu:0.5)|||}, {|kitchen|bed room|garden|cosmic dust|cyber punk city|white|simple} background, model photoshot, fashion photoshot, highly detailed, 4k, high resolution"
# g_template = 'gold (glasses:1.4)'
# prrompt_template = "lolita girl"
folder_save = "/content/drive/MyDrive/outputs/"
folder_name = "i2i-dwrt-pulid-tcd"
output_path=f"{folder_save}{folder_name}-[time(%Y-%m-%d-%H)]"
i2i_path = "/content/i2i-768/"
ref_path = "/content/pulid-colab-1/"
wm_folder = "/content/colab/wildcard"
num_images = 20
repeat_img = 5
meg_weight = 0

# ckpt_name="RealVisXL_V4.0.safetensors"
# ckpt_name="RRealistic_Stock_Photo_v2.safetensors"
ckpt_name="samaritan.safetensors"
img_ref_folder = (

    (
        f'{ref_path}01.jpg',
        f'{ref_path}02.jpg',
        f'{ref_path}03.jpg',
        f'{ref_path}04.jpg',
    ),

    (
        f'{ref_path}01.jpg',
        f'{ref_path}02.jpg',
        f'{ref_path}03.jpg',
        f'{ref_path}04.jpg',
    ),
    (
        f'{ref_path}97-p-02.jpg',
        f'{ref_path}97-p-01.jpg',
        f'{ref_path}97-p-03.jpg',
        f'{ref_path}97-p-04.jpg',
    ),
    (
        f'{ref_path}97-m-03.jpg',
        f'{ref_path}97-m-02.jpg',
        f'{ref_path}97-m-01.jpg',
        f'{ref_path}97-m-04.jpg',
    ),


)


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
    from nodes import init_external_custom_nodes, init_builtin_extra_nodes, init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_external_custom_nodes()
    init_builtin_extra_nodes()
    init_extra_nodes()


from nodes import (
    CheckpointLoaderSimple,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    LoraLoader,
    VAEEncode,
    LoadImage,
    ConditioningConcat,
    VAEDecode,
    VAELoader,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        vaeloader = VAELoader()
        vaeloader_10 = vaeloader.load_vae(vae_name="sdxl_vae.safetensors")

        # emptylatentimage = EmptyLatentImage()
        # emptylatentimage_5 = emptylatentimage.generate(
        #     width=576, height=1024, batch_size=1
        # )

        loraloader = LoraLoader()
        loraloader_40 = loraloader.load_lora(
            # lora_name="sdxl_meg-000008.safetensors",
            lora_name="sdxl_meg-240628-000020.safetensors",
            strength_model=meg_weight,
            strength_clip=meg_weight,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloader_39 = loraloader.load_lora(
            lora_name="DetailTweakerXL.safetensors",
            # lora_name="extreamly-detailed.safetensors",
            strength_model=3,
            strength_clip=3,
            model=get_value_at_index(loraloader_40, 0),
            clip=get_value_at_index(loraloader_40, 1),
        )

        loraloader_38 = loraloader.load_lora(
            lora_name="Hyper-SDXL-8steps-lora.safetensors",
            strength_model=0.9500000000000001,
            strength_clip=0.9500000000000001,
            model=get_value_at_index(loraloader_39, 0),
            clip=get_value_at_index(loraloader_39, 1),
        )

        cliptextencode = CLIPTextEncode()


        pulidmodelloader = NODE_CLASS_MAPPINGS["PulidModelLoader"]()
        pulidmodelloader_16 = pulidmodelloader.load_model(
            pulid_file="ip-adapter_pulid_sdxl_fp16.safetensors"
        )

        pulidevacliploader = NODE_CLASS_MAPPINGS["PulidEvaClipLoader"]()
        pulidevacliploader_17 = pulidevacliploader.load_eva_clip()

        pulidinsightfaceloader = NODE_CLASS_MAPPINGS["PulidInsightFaceLoader"]()
        pulidinsightfaceloader_18 = pulidinsightfaceloader.load_insightface(
            provider="CPU"
        )



   

        imagebatchmultiple = NODE_CLASS_MAPPINGS["ImageBatchMultiple+"]()
        applypulidadvanced = NODE_CLASS_MAPPINGS["ApplyPulidAdvanced"]()
        tcdmodelsamplingdiscrete = NODE_CLASS_MAPPINGS["TCDModelSamplingDiscrete"]()
        conditioningconcat = ConditioningConcat()
        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()
        image_load = NODE_CLASS_MAPPINGS["Image Load"]()
        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()
        vaeencode = VAEEncode()
        image_resize = NODE_CLASS_MAPPINGS["Image Resize"]()

        for p1,p2,p3,p4 in img_ref_folder:
            wm = WildcardManager(Path(wm_folder))
            generator = RandomPromptGenerator(wildcard_manager=wm)
            prompts = list(generator.generate(prrompt_template, num_images=num_images))

            
            
            image_load_57 = image_load.load_image(
                image_path=p4,
                RGBA="false",
                filename_text_extension="true",
            )

            image_load_58 = image_load.load_image(
                image_path=p1,
                RGBA="false",
                filename_text_extension="true",
            )

            image_load_59 = image_load.load_image(
                image_path=p2,
                RGBA="false",
                filename_text_extension="true",
            )

            image_load_60 = image_load.load_image(
                image_path=p3,
                RGBA="false",
                filename_text_extension="true",
            )

            imagebatchmultiple_50 = imagebatchmultiple.execute(
                method="lanczos",
                image_1=get_value_at_index(image_load_57, 0),
                image_2=get_value_at_index(image_load_58, 0),
                image_3=get_value_at_index(image_load_59, 0),
                image_4=get_value_at_index(image_load_60, 0),
            )

            applypulidadvanced_22 = applypulidadvanced.apply_pulid(
                weight=0.75,
                projection="ortho_v2",
                fidelity=2,
                noise=0.7000000000000001,
                start_at=0,
                end_at=1,
                model=get_value_at_index(loraloader_38, 0),
                pulid=get_value_at_index(pulidmodelloader_16, 0),
                eva_clip=get_value_at_index(pulidevacliploader_17, 0),
                face_analysis=get_value_at_index(pulidinsightfaceloader_18, 0),
                image=get_value_at_index(imagebatchmultiple_50, 0),
            )



            for q in range(repeat_img):
                load_image_batch_61 = load_image_batch.load_batch_images(
                    mode="random",
                    index=0,
                    label="Batch 001",
                    path=i2i_path,
                    pattern="*",
                    allow_RGBA_output="false",
                    filename_text_extension="true",
                )

                
                # image_resize_60 = image_resize.image_rescale(
                #     mode="rescale",
                #     supersample="true",
                #     resampling="lanczos",
                #     rescale_factor=1,
                #     resize_width=584,
                #     resize_height=1536,
                #     image=get_value_at_index(load_image_batch_61, 0),
                # )

                
                vaeencode_59 = vaeencode.encode(
                    pixels=get_value_at_index(load_image_batch_61, 0),
                    vae=get_value_at_index(vaeloader_10, 0),
                )



                for prompt in prompts:

                    denoise = random.choice([0.75,0.65,0.7])

                    tcdmodelsamplingdiscrete_12 = tcdmodelsamplingdiscrete.patch(
                        steps=8,
                        scheduler="sgm_uniform",
                        denoise=denoise,
                        eta=0.8,
                        model=get_value_at_index(applypulidadvanced_22, 0),
                    )


                    cliptextencode_6 = cliptextencode.encode(
                        text=prompt,
                        clip=get_value_at_index(loraloader_38, 1),
                    )
                    
                    g_prompt = list(generator.generate(g_template, num_images=1))

                    cliptextencode_55 = cliptextencode.encode(
                        text=g_prompt[0], 
                        clip=get_value_at_index(loraloader_38, 1)
                    )

                    cliptextencode_7 = cliptextencode.encode(
                        text="blurry, malformed, low quality, worst quality, artifacts, noise, text, watermark, glitch, deformed, ugly, horror, ill",
                        clip=get_value_at_index(loraloader_38, 1),
                    )

                    conditioningconcat_56 = conditioningconcat.concat(
                        conditioning_to=get_value_at_index(cliptextencode_6, 0),
                        conditioning_from=get_value_at_index(cliptextencode_55, 0),
                    )






                    samplercustom_11 = samplercustom.sample(
                        add_noise=True,
                        noise_seed=random.randint(1, 2**64),
                        cfg=1,
                        model=get_value_at_index(tcdmodelsamplingdiscrete_12, 0),
                        positive=get_value_at_index(conditioningconcat_56, 0),
                        negative=get_value_at_index(cliptextencode_7, 0),
                        sampler=get_value_at_index(tcdmodelsamplingdiscrete_12, 1),
                        sigmas=get_value_at_index(tcdmodelsamplingdiscrete_12, 2),
                        latent_image=get_value_at_index(vaeencode_59, 0),
                    )

                    vaedecode_8 = vaedecode.decode(
                        samples=get_value_at_index(samplercustom_11, 0),
                        vae=get_value_at_index(vaeloader_10, 0),
                    )

                    image_save_14 = image_save.was_save_images(
                        output_path=output_path,
                        # output_path="pulid-meg",
                        filename_prefix=f"{clean(prompt)}-{clean(g_prompt[0])}-de-{denoise}",
                        # filename_prefix="pulid-meg",
                        filename_delimiter="_",
                        filename_number_padding=4,
                        filename_number_start="false",
                        extension="webp",
                        quality=80,
                        lossless_webp="false",
                        overwrite_mode="true",
                        show_history="false",
                        show_history_by_prefix="false",
                        embed_workflow="false",
                        show_previews="false",
                        images=get_value_at_index(vaedecode_8, 0),
                    )


if __name__ == "__main__":
    main()
