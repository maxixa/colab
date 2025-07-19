import os
import random
import re
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager
import textwrap

g_template = "{red|pink|white|gold|blue|black|silver|gold} {(round glasses:1.3)|glasses} "
# prrompt_template = "{close up|}, {candid photography|elegance|fashion|fashion photography|stylish|casual style|cinematic} photo of woman, {||twintails}, {blonde|}, {bow hair|cat ears|Bows|Hair Clips|Headbands|Hair Ties|Barrettes|Hair Slides|Ponytail Holders|Hair Pins|Flower Crowns|Bobby Pins|Hair Sticks|Hair Combs|Scrunchies|Hair Tassels|Crown Headbands|Hair Charms|Braided Headbands|Hair Wraps|Ponytail Streamers|Glitter Hair Ties}, {(sexy:1.3)|}{|pink|red|peach|maroon|light-blue|Navy|Scarlet|Royal-blue|Turquoise|Olive|Emerald|Sage|Gold|Cream|Purple|Lavender|Violet|Brown|Tan|Blush|Rose|Fuchsia|Magenta|pink||} {lolita dress|fairy dress, wings|princess dress|ballgown|wedding dress| BUTTERFLY DRESS, wings|BURLESQUE DRESS|CUTE mini DRESS|FLOWER DRESS|SAILOR SENSHI UNIFORM| VICTORIAN DRESS| VICTORIAN mini DRESS} ,{||white thighhighs||white stockings|}, {Sashes|Ruffles|Bows|Ribbons|Lace Trims|Petticoats|Tutus|Belts|Buckles|Brooches|Flower Pins|Appliques|Embroidery|Patches|Ribbon Bows|Dress Clips|Waist Belts|Dress Pins|Dress Brooches|Dress Sashes}, {(tutu:0.7)|(tutu:0.5)|||}, {depth of field|kitchen|white|blurred|indoor|white|bokeh} background, {elegance|model photoshot}, {fashion|fashion photography},  dynamic pose, {high-resolution image-|high-resolution}"
prrompt_template = "{full body|midshot|full body, high hells}, {candid photography|elegance|fashion|fashion photography|stylish|casual style|cinematic} photo of woman, {Over-the-Shoulder Look| Leaning Against a Wall| Sitting on the Ground| Walking Toward the Camera| Arms Crossed| Hands on Hips| Lean Forward| One Hand on Chin| Twirling Hair| Jumping in the Air| Hand in Hair| Crossed Legs, Standing| Hands Behind Back| Laying on Stomach| Sitting | Touching Face Lightly| Stretching Arms Upward| Holding a Prop| Leaning Forward with Elbows on Knees| Hands Clasped Together| Kicking One Leg Up| One Foot Forward| Hugging Self| Resting Head on Hand| Hands Resting on Lap| Mid-Spin with Dress|sitting pose|laying pose|kneeling pose|sitting pose|laying pose|kneeling pose}, {||twintails}, {standing pose|dynamic pose|sitting pose|kneeling pose|dancing pose}, {blonde|}, {bow hair|cat ears|Bows|Hair Clips|Headbands|Hair Ties|Barrettes|Hair Slides|Ponytail Holders|Hair Pins|Flower Crowns|Bobby Pins|Hair Sticks|Hair Combs|Scrunchies|Hair Tassels|Crown Headbands|Hair Charms|Braided Headbands|Hair Wraps|Ponytail Streamers|Glitter Hair Ties}, {(sexy:1.3)|}{|pink|red|peach|maroon|light-blue|Navy|Scarlet|Royal-blue|Turquoise|Olive|Emerald|Sage|Gold|Cream|Purple|Lavender|Violet|Brown|Tan|Blush|Rose|Fuchsia|Magenta|pink||} {lolita dress|fairy dress, wings|princess dress|ballgown|wedding dress| BUTTERFLY DRESS, wings|BURLESQUE DRESS|CUTE mini DRESS|FLOWER DRESS|SAILOR SENSHI UNIFORM| VICTORIAN DRESS| VICTORIAN mini DRESS} ,{||white thighhighs||white stockings|}, {Sashes|Ruffles|Bows|Ribbons|Lace Trims|Petticoats|Tutus|Belts|Buckles|Brooches|Flower Pins|Appliques|Embroidery|Patches|Ribbon Bows|Dress Clips|Waist Belts|Dress Pins|Dress Brooches|Dress Sashes}, {(tutu:0.7)|(tutu:0.5)|||}, {depth of field|kitchen|garden|blurred|indoor|white|bokeh} background, {elegance|model photoshot}, {fashion|fashion photography},  dynamic pose, {high-resolution image-|high-resolution}"
# g_template = 'gold (glasses:1.4)'
# prrompt_template = "lolita girl"
folder_save = "/content/drive/MyDrive/outputs/"
folder_name = "dwrt-photomaker-tcd"
output_path=f"{folder_save}{folder_name}-[time(%Y-%m-%d-%H)]"
ref_path = "/content/pulid-colab-1/"
wm_folder = "/content/colab/wildcard"
num_images = 1000
meg_weight = 0.8

lora_name_1="Hyper-SDXL-8steps-lora.safetensors"
strength_model_1=0.85

lora_name_2="sdxl_meg-240628-000020.safetensors"
# lora_name_2="sdxl_meg-metal-240831.safetensors"
# lora_name_2="sdxl_meg-metal-240831-000020.safetensors"
strength_model_2=0.75

lora_name_3="photomaker-v2.bin"
strength_model_3=0.85

# ckpt_name="RealVisXL_V4.0.safetensors"
# ckpt_name="RealVisXL_V5.0_fp16.safetensors"
# ckpt_name="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
# ckpt_name="Juggernaut-X-RunDiffusion-NSFW.safetensors"
# ckpt_name="ProteusV0.4-RunDiffusionPhoto.safetensors"
# ckpt_name="samaritan.safetensors"
# ckpt_name="Realistic_Stock_Photo_v2.safetensors"
# ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
# ckpt_name="Colossus_Project_X_Midgard.SafeTensors"
# ckpt_name="Colossus_Project_XL_NEO_10B.SafeTensors"
# ckpt_name="Colossus_Project_XL_12C.SafeTensors"
# ckpt_name="NightVisionXL-9.safetensors"
ckpt_name="Realism_Engine_SDXL_3.safetensors"
# ckpt_name=""

img_ref_folder = (
  (
    f'{ref_path}01.jpg',
    f'{ref_path}02.jpg',
    f'{ref_path}03.jpg',
    f'{ref_path}04.jpg',
  ),
  (
    f'{ref_path}02.jpg',
    f'{ref_path}01.jpg',
    f'{ref_path}03.jpg',
    f'{ref_path}04.jpg',
  ),
  (
    f'{ref_path}03.jpg',
    f'{ref_path}04.jpg',
    f'{ref_path}01.jpg',
    f'{ref_path}02.jpg',
  ),
  # (
  #     f'{ref_path}04-03.jpg',
  #     f'{ref_path}04-02.jpg',
  #     f'{ref_path}04-01.jpg',
  #     f'{ref_path}04-04.jpg',
  # ),
  # (
  #     f'{ref_path}16-03.jpg',
  #     f'{ref_path}16-02.jpg',
  #     f'{ref_path}16-01.jpg',
  #     f'{ref_path}16-04.jpg',
  # ),
  # (
  #     f'{ref_path}97-m-03.jpg',
  #     f'{ref_path}97-m-02.jpg',
  #     f'{ref_path}97-m-01.jpg',
  #     f'{ref_path}97-m-04.jpg',
  # ),
  # (
  #     f'{ref_path}97-p-03.jpg',
  #     f'{ref_path}97-p-02.jpg',
  #     f'{ref_path}97-p-01.jpg',
  #     f'{ref_path}97-p-04.jpg',
  # ),
  (
    f'{ref_path}01.jpg',
    f'{ref_path}02.jpg',
    f'{ref_path}03.jpg',
    f'{ref_path}04.jpg',
  ),
    (
    f'{ref_path}04.jpg',
    f'{ref_path}03.jpg',
    f'{ref_path}01.jpg',
    f'{ref_path}02.jpg',
  ),
  (
    f'{ref_path}01.jpg',
    f'{ref_path}02.jpg',
    f'{ref_path}03.jpg',
    f'{ref_path}04.jpg',
  ),

  #   (
  #   f'{ref_path}22-04.jpg',
  #   f'{ref_path}22-02.jpg',
  #   f'{ref_path}22-03.jpg',
  #   f'{ref_path}22-01.jpg',
  # ),


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
    from nodes import (
        init_external_custom_nodes,
        init_builtin_extra_nodes,
        init_extra_nodes,
    )
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
    ConditioningConcat,
    VAELoader,
    CLIPTextEncode,
    LoraLoaderModelOnly,
    VAEEncode,
    NODE_CLASS_MAPPINGS,
    CheckpointLoaderSimple,
    VAEDecode,
    EmptyLatentImage,
    CLIPSetLastLayer,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_10 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2, clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=640, height=1136, batch_size=1
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_7 = cliptextencode.encode(
            text="asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loraloadermodelonly = LoraLoaderModelOnly()
        loraloadermodelonly_80 = loraloadermodelonly.load_lora_model_only(
            lora_name=lora_name_1,
            strength_model=strength_model_1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
        )

        loraloadermodelonly_89 = loraloadermodelonly.load_lora_model_only(
            lora_name=lora_name_2,
            strength_model=strength_model_2,
            model=get_value_at_index(loraloadermodelonly_80, 0),
        )

        # loraloadermodelonly_53 = loraloadermodelonly.load_lora_model_only(
        #     lora_name=lora_name_3,
        #     strength_model=strength_model_3,
        #     model=get_value_at_index(loraloadermodelonly_89, 0),
        # )

        photomakerloaderplus = NODE_CLASS_MAPPINGS["PhotoMakerLoaderPlus"]()
        photomakerloaderplus_74 = photomakerloaderplus.load_photomaker_model(
            photomaker_model_name="photomaker-v2.bin"
        )

        photomakerloraloaderplus = NODE_CLASS_MAPPINGS["PhotoMakerLoraLoaderPlus"]()
        photomakerloraloaderplus_11 = photomakerloraloaderplus.load_photomaker_lora(
            lora_strength=strength_model_3,
            model=get_value_at_index(loraloadermodelonly_89, 0),
            photomaker=get_value_at_index(photomakerloaderplus_74, 0),
        )

        photomakerinsightfaceloader = NODE_CLASS_MAPPINGS[
            "PhotoMakerInsightFaceLoader"
        ]()
        photomakerinsightfaceloader_76 = photomakerinsightfaceloader.load_insightface(
            provider="CPU"
        )

        vaeloader = VAELoader()
        vaeloader_77 = vaeloader.load_vae(vae_name="sdxl_vae.safetensors")

        # load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()
        # load_image_batch_88 = load_image_batch.load_batch_images(
        #     mode="incremental_image",
        #     index=0,
        #     label="Batch 001",
        #     path="output/dwrt-pulid-tcd-2024-07-29-07",
        #     pattern="*",
        #     allow_RGBA_output="false",
        #     filename_text_extension="true",
        # )

        # vaeencode = VAEEncode()
        # vaeencode_85 = vaeencode.encode(
        #     pixels=get_value_at_index(load_image_batch_88, 0),
        #     vae=get_value_at_index(vaeloader_77, 0),
        # )

        image_load = NODE_CLASS_MAPPINGS["Image Load"]()




        tcdmodelsamplingdiscrete = NODE_CLASS_MAPPINGS["TCDModelSamplingDiscrete"]()
        imagebatchmultiple = NODE_CLASS_MAPPINGS["ImageBatchMultiple+"]()
        photomakerencodeplus = NODE_CLASS_MAPPINGS["PhotoMakerEncodePlus"]()
        conditioningconcat = ConditioningConcat()
        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        tcdmodelsamplingdiscrete_82 = tcdmodelsamplingdiscrete.patch(
            steps=10,
            scheduler="sgm_uniform",
            denoise=1,
            eta=0.1,
            model=get_value_at_index(photomakerloraloaderplus_11, 0),
        )

        for p1,p2,p3,p4 in img_ref_folder:
            wm = WildcardManager(Path(wm_folder))
            generator = RandomPromptGenerator(wildcard_manager=wm)
            prompts = list(generator.generate(prrompt_template, num_images=num_images))

            image_load_94 = image_load.load_image(
                image_path=p1, RGBA="false", filename_text_extension="true"
            )

            image_load_95 = image_load.load_image(
                image_path=p2, RGBA="false", filename_text_extension="true"
            )

            image_load_96 = image_load.load_image(
                image_path=p3, RGBA="false", filename_text_extension="true"
            )

            image_load_97 = image_load.load_image(
                image_path=p4, RGBA="false", filename_text_extension="true"
            )


            imagebatchmultiple_78 = imagebatchmultiple.execute(
                method="lanczos",
                image_1=get_value_at_index(image_load_94, 0),
                image_2=get_value_at_index(image_load_95, 0),
                image_3=get_value_at_index(image_load_96, 0),
                image_4=get_value_at_index(image_load_97, 0),
            )

            photomakerencodeplus_73 = photomakerencodeplus.apply_photomaker(
                trigger_word="img",
                text="a woman img ,  ",
                clip=get_value_at_index(clipsetlastlayer_10, 0),
                photomaker=get_value_at_index(photomakerloaderplus_74, 0),
                image=get_value_at_index(imagebatchmultiple_78, 0),
                insightface_opt=get_value_at_index(photomakerinsightfaceloader_76, 0),
            )

            for prompt in prompts:

                cliptextencode_99 = cliptextencode.encode(
                    text=prompt,
                    clip=get_value_at_index(checkpointloadersimple_4, 1),
                )

                g_prompt = list(generator.generate(g_template, num_images=1))

                cliptextencode_101 = cliptextencode.encode(
                    text=g_prompt[0], clip=get_value_at_index(checkpointloadersimple_4, 1)
                )

                conditioningconcat_102 = conditioningconcat.concat(
                    conditioning_to=get_value_at_index(cliptextencode_99, 0),
                    conditioning_from=get_value_at_index(cliptextencode_101, 0),
                )

                conditioningconcat_98 = conditioningconcat.concat(
                    conditioning_to=get_value_at_index(photomakerencodeplus_73, 0),
                    conditioning_from=get_value_at_index(conditioningconcat_102, 0),
                )

                samplercustom_81 = samplercustom.sample(
                    add_noise=True,
                    noise_seed=random.randint(1, 2**64),
                    cfg=1,
                    model=get_value_at_index(tcdmodelsamplingdiscrete_82, 0),
                    positive=get_value_at_index(conditioningconcat_98, 0),
                    negative=get_value_at_index(cliptextencode_7, 0),
                    sampler=get_value_at_index(tcdmodelsamplingdiscrete_82, 1),
                    sigmas=get_value_at_index(tcdmodelsamplingdiscrete_82, 2),
                    latent_image=get_value_at_index(emptylatentimage_5, 0),
                )

                vaedecode_8 = vaedecode.decode(
                    samples=get_value_at_index(samplercustom_81, 0),
                    vae=get_value_at_index(vaeloader_77, 0),
                )

                image_save_79 = image_save.was_save_images(
                    output_path=f"{output_path}/{ckpt_name}-{strength_model_3}",
                    filename_prefix=f"{clean(textwrap.shorten(prompt, width=180))}-{ckpt_name}",
                    filename_delimiter="_",
                    filename_number_padding=4,
                    filename_number_start="false",
                    extension="webp",
                    dpi=300,
                    quality=80,
                    optimize_image="true",
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
