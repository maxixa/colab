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

g_template = "{red|pink|white|gold|silver} (glasses:1)"
prrompt_template = "cinematic, glamour (full body:1) photo of woman, {||twintails}, {blonde|}, glasses, {bow hair|cat ears|Bows|Hair Clips|Headbands|Hair Ties|Barrettes|Hair Slides|Ponytail Holders|Hair Pins|Flower Crowns|Bobby Pins|Hair Sticks|Hair Combs|Scrunchies|Hair Tassels|Crown Headbands|Hair Charms|Braided Headbands|Hair Wraps|Ponytail Streamers|Glitter Hair Ties}, {|pink|red|peach|maroon|light-blue|Navy|Scarlet|Royal-blue|Turquoise|Olive|Emerald|Sage|Gold|Cream|Purple|Lavender|Violet|Brown|Tan|Blush|Rose|Fuchsia|Magenta|pink||} {lolita dress|fairy dress, wings|princess dress|ballgown|wedding dress| BUTTERFLY DRESS, wings|BURLESQUE DRESS|CUTE mini DRESS|FLOWER DRESS|SAILOR SENSHI UNIFORM| VICTORIAN DRESS| VICTORIAN mini DRESS} ,{||white thighhighs||white stockings|}, {Sashes|Ruffles|Bows|Ribbons|Lace Trims|Petticoats|Tutus|Belts|Buckles|Brooches|Flower Pins|Appliques|Embroidery|Patches|Ribbon Bows|Dress Clips|Waist Belts|Dress Pins|Dress Brooches|Dress Sashes}, {(tutu:0.7)|(tutu:0.5)|||}, {|kitchen|bed room|garden|cosmic dust|cyber punk city|white|simple} background, model photoshot, fashion photoshot, highly detailed, 4k, high resolution"
folder_save = ""
folder_name = "dwrt-inpaint-tcd-SDXL"
output_path=f"{folder_save}{folder_name}-[time(%Y-%m-%d-%H)]"
ref_path = "/content/pulid-colab-1/"
wm_folder = "wildcard"
num_images = 1 #number of prompt generated
num_que = 2 #number of general repeat

width=576
height=1024

# ckpt_name="Deliberate_v6.safetensors"
# ckpt_name="RealVisV4-meg-out-all.safetensors"
# ckpt_name="Realistic_Stock_Photo_v2.safetensors"
# ckpt_name="RealisticVisionV51.safetensors"
# ckpt_name="Aardvark-2024-Photography.safetensors"
# ckpt_name="Reliberate.safetensors"
# ckpt_name="majicMIXrealistic.safetensors"
# ckpt_name="epiCRealism.safetensors"
# ckpt_name="Aardvark-2024-Photography.safetensors"
ckpt_name="RealVisXL_V4.0.safetensors"

# lora_name_1 = "All-in-one-Clothing-Bundle-SFW.SafeTensors"
# lora_name_1 = "CopaxDress.safetensors"
lora_name_1 = "Hyper-SDXL-8steps-lora.safetensors"
strength_model_1 = 0

lora_name_2 = "Hyper-SDXL-8steps-lora.safetensors"
strength_model_2 = 0.9

lora_name_3 = "Hyper-SDXL-8steps-lora.safetensors"
strength_model_3 = 0

control_net_name="controlnet_plus_promax.safetensors"




img_mask_folder = (

  # ('/home/studio-lab-user/masking/89/300/img','/home/studio-lab-user/masking/89/300/mask','png'),
  # ('/home/studio-lab-user/masking/89/350/img','/home/studio-lab-user/masking/89/350/mask','png'),
  # ('/home/studio-lab-user/masking/89/400/img','/home/studio-lab-user/masking/89/400/mask','png'),
  # ('/home/studio-lab-user/masking/89/450/img','/home/studio-lab-user/masking/89/450/mask','png'),
  # ('/home/studio-lab-user/masking/89/500/img','/home/studio-lab-user/masking/89/500/mask','png'),
  # ('/home/studio-lab-user/masking/89/550/img','/home/studio-lab-user/masking/89/550/mask','png'),
  # ('/home/studio-lab-user/masking/89/600/img','/home/studio-lab-user/masking/89/600/mask','png'),

  # ('/home/studio-lab-user/masking/93-mask/300/img','/home/studio-lab-user/masking/93-mask/300/mask','png'),
  # ('/home/studio-lab-user/masking/93-mask/350/img','/home/studio-lab-user/masking/93-mask/350/mask','png'),
  # ('/home/studio-lab-user/masking/93-mask/400/img','/home/studio-lab-user/masking/93-mask/400/mask','png'),
  # ('/home/studio-lab-user/masking/93-mask/450/img','/home/studio-lab-user/masking/93-mask/450/mask','png'),
  # ('/home/studio-lab-user/masking/93-mask/500/img','/home/studio-lab-user/masking/93-mask/500/mask','png'),
  # ('/home/studio-lab-user/masking/93-mask/550/img','/home/studio-lab-user/masking/93-mask/550/mask','png'),
  # ('/home/studio-lab-user/masking/93-mask/600/img','/home/studio-lab-user/masking/93-mask/600/mask','png'),

  # ('/home/studio-lab-user/masking/94-mask/300/img','/home/studio-lab-user/masking/94-mask/300/mask','png'),
  # ('/home/studio-lab-user/masking/94-mask/350/img','/home/studio-lab-user/masking/94-mask/350/mask','png'),
  # ('/home/studio-lab-user/masking/94-mask/400/img','/home/studio-lab-user/masking/94-mask/400/mask','png'),
  # ('/home/studio-lab-user/masking/94-mask/450/img','/home/studio-lab-user/masking/94-mask/450/mask','png'),
  # ('/home/studio-lab-user/masking/94-mask/500/img','/home/studio-lab-user/masking/94-mask/500/mask','png'),
  # ('/home/studio-lab-user/masking/94-mask/550/img','/home/studio-lab-user/masking/94-mask/550/mask','png'),
  # ('/home/studio-lab-user/masking/94-mask/600/img','/home/studio-lab-user/masking/94-mask/600/mask','png'),

)

img_mask_folder = (
  ('/home/studio-lab-user/masking/94-mask/250/img','/home/studio-lab-user/masking/94-mask/250/mask','png'),
  ('/home/studio-lab-user/masking/94-mask/250/img-fliped','/home/studio-lab-user/masking/94-mask/250/mask-fliped','png'),
  ('/home/studio-lab-user/masking/93-mask/250/img','/home/studio-lab-user/masking/93-mask/250/mask','png'),
  ('/home/studio-lab-user/masking/93-mask/250/img-fliped','/home/studio-lab-user/masking/93-mask/250/mask-fliped','png'),
  ('/home/studio-lab-user/masking/89/250/img','/home/studio-lab-user/masking/89/250/mask','png'),
  ('/home/studio-lab-user/masking/89/250/img-fliped','/home/studio-lab-user/masking/89/250/mask-fliped','png'),
)



# img_mask_folder = (
#   ('/home/studio-lab-user/masking/89/250/img','/home/studio-lab-user/masking/89/250/mask','png'),
#   ('/home/studio-lab-user/masking/89/300/img','/home/studio-lab-user/masking/89/300/mask','png'),
#   ('/home/studio-lab-user/masking/89/350/img','/home/studio-lab-user/masking/89/350/mask','png'),
#   ('/home/studio-lab-user/masking/89/400/img','/home/studio-lab-user/masking/89/400/mask','png'),
#   ('/home/studio-lab-user/masking/89/450/img','/home/studio-lab-user/masking/89/450/mask','png'),
#   ('/home/studio-lab-user/masking/89/500/img','/home/studio-lab-user/masking/89/500/mask','png'),
#   ('/home/studio-lab-user/masking/89/250/img-fliped','/home/studio-lab-user/masking/89/250/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/89/300/img-fliped','/home/studio-lab-user/masking/89/300/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/89/350/img-fliped','/home/studio-lab-user/masking/89/350/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/89/400/img-fliped','/home/studio-lab-user/masking/89/400/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/89/450/img-fliped','/home/studio-lab-user/masking/89/450/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/89/500/img-fliped','/home/studio-lab-user/masking/89/500/mask-fliped','png'),
# )

#@title ## Body mask 93 94
# img_mask_folder = (
#   ('/home/studio-lab-user/masking/93-body-v2-mask/600/img','/home/studio-lab-user/masking/93-body-v2-mask/600/mask','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/600/img','/home/studio-lab-user/masking/94-body-v2-mask/600/mask','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/600/img','/home/studio-lab-user/masking/94-body-v2-mask/600/mask','png'),
#   ('/home/studio-lab-user/masking/93-body-v2-mask/650/img','/home/studio-lab-user/masking/93-body-v2-mask/650/mask','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/650/img','/home/studio-lab-user/masking/94-body-v2-mask/650/mask','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/650/img','/home/studio-lab-user/masking/94-body-v2-mask/650/mask','png'),
#   ('/home/studio-lab-user/masking/93-body-v2-mask/550/img','/home/studio-lab-user/masking/93-body-v2-mask/550/mask','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/550/img','/home/studio-lab-user/masking/94-body-v2-mask/550/mask','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/550/img','/home/studio-lab-user/masking/94-body-v2-mask/550/mask','png'),
#   ('/home/studio-lab-user/masking/93-body-v2-mask/550/img-fliped','/home/studio-lab-user/masking/93-body-v2-mask/550/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/550/img-fliped','/home/studio-lab-user/masking/94-body-v2-mask/550/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/550/img-fliped','/home/studio-lab-user/masking/94-body-v2-mask/550/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/93-body-v2-mask/650/img-fliped','/home/studio-lab-user/masking/93-body-v2-mask/650/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/650/img-fliped','/home/studio-lab-user/masking/94-body-v2-mask/650/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/650/img-fliped','/home/studio-lab-user/masking/94-body-v2-mask/650/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/93-body-v2-mask/600/img-fliped','/home/studio-lab-user/masking/93-body-v2-mask/600/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/600/img-fliped','/home/studio-lab-user/masking/94-body-v2-mask/600/mask-fliped','png'),
#   ('/home/studio-lab-user/masking/94-body-v2-mask/600/img-fliped','/home/studio-lab-user/masking/94-body-v2-mask/600/mask-fliped','png'),
# )


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
    NODE_CLASS_MAPPINGS,
    ConditioningConcat,
    VAEDecode,
    VAELoader,
    CLIPTextEncode,
    EmptyLatentImage,
    CheckpointLoaderSimple,
    ControlNetLoader,
    ControlNetApplyAdvanced,
    LoraLoader,
    CLIPSetLastLayer,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()
        cliptextencode = CLIPTextEncode()

        checkpointloadersimple_2 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt_name
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_10 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2, clip=get_value_at_index(checkpointloadersimple_2, 1)
        )

        loraloader = LoraLoader()
        loraloader_24 = loraloader.load_lora(
            lora_name=lora_name_1,
            strength_model=strength_model_1,
            strength_clip=strength_model_1,
            model=get_value_at_index(checkpointloadersimple_2, 0),
            clip=get_value_at_index(clipsetlastlayer_10, 1),
        )



        loraloader_26 = loraloader.load_lora(
            lora_name=lora_name_3,
            strength_model=strength_model_3,
            strength_clip=strength_model_3,
            model=get_value_at_index(loraloader_24, 0),
            clip=get_value_at_index(loraloader_24, 1),
        )

        loraloader_25 = loraloader.load_lora(
            lora_name=lora_name_2,
            strength_model=strength_model_2,
            strength_clip=strength_model_2,
            model=get_value_at_index(loraloader_26, 0),
            clip=get_value_at_index(loraloader_26, 1),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_9 = controlnetloader.load_controlnet(
            control_net_name=control_net_name
        )

        setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()
        setunioncontrolnettype_30 = setunioncontrolnettype.set_controlnet_type(
            type="repaint", control_net=get_value_at_index(controlnetloader_9, 0)
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_24 = emptylatentimage.generate(
            width=width, height=height, batch_size=1
        )

        vaeloader = VAELoader()
        vaeloader_27 = vaeloader.load_vae(vae_name="sdxl_vae.safetensors")

        tcdmodelsamplingdiscrete = NODE_CLASS_MAPPINGS["TCDModelSamplingDiscrete"]()
        conditioningconcat = ConditioningConcat()
        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        maskfromrgbcmybw = NODE_CLASS_MAPPINGS["MaskFromRGBCMYBW+"]()
        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        inpaintpreprocessor = NODE_CLASS_MAPPINGS["InpaintPreprocessor"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        vaedecode = VAEDecode()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()
        image_load = NODE_CLASS_MAPPINGS["Image Load"]()
        images_to_rgb = NODE_CLASS_MAPPINGS["Images to RGB"]()
        setunioncontrolnettype = NODE_CLASS_MAPPINGS["SetUnionControlNetType"]()

        tcdmodelsamplingdiscrete_72 = tcdmodelsamplingdiscrete.patch(
            steps=8,
            scheduler="sgm_uniform",
            denoise=1,
            eta=0.1,
            model=get_value_at_index(loraloader_25, 0),
        )

        for q in range(num_que):



            # load_image_batch_52 = load_image_batch.load_batch_images(
            #     mode="incremental_image",
            #     index=0,
            #     label="Batch 002",
            #     path="MASK/93-mask/350/mask",
            #     pattern="*",
            #     allow_RGBA_output="true",
            #     filename_text_extension="true",
            # )

            # load_image_batch_53 = load_image_batch.load_batch_images(
            #     mode="incremental_image",
            #     index=0,
            #     label="Batch 001",
            #     path="MASK/93-mask/350/img",
            #     pattern="*",
            #     allow_RGBA_output="false",
            #     filename_text_extension="true",
            # )

            for img, mask, ext in img_mask_folder:

              image_folder = img
              mask_folder = mask
              mask_ext = ext
              parts = image_folder.split('/')
              num = parts[-2]

              image_files = os.listdir(image_folder)
              mask_files = os.listdir(mask_folder)

              for image_file in image_files:
                name, ext = os.path.splitext(image_file)
                if f'{name}.{mask_ext}' in mask_files:
                  image_path = os.path.join(image_folder, image_file)
                  mask_path = os.path.join(mask_folder, name + '.' + mask_ext)

                  wm = WildcardManager(Path(wm_folder))
                  generator = RandomPromptGenerator(wildcard_manager=wm)
                  prompts = list(generator.generate(prrompt_template, num_images=num_images))

                  cliptextencode_45 = cliptextencode.encode(
                      text=prompts[0],
                      clip=get_value_at_index(loraloader_25, 1),
                  )

                  cliptextencode_4 = cliptextencode.encode(
                      text="0", clip=get_value_at_index(loraloader_25, 1)
                  )

                  cliptextencode_5 = cliptextencode.encode(
                      text="black and white", clip=get_value_at_index(checkpointloadersimple_2, 1)
                  )

                  image_load_57 = image_load.load_image(
                      image_path=image_path,
                      RGBA="false",
                      filename_text_extension="true",
                  )

                  image_load_58 = image_load.load_image(
                      image_path=mask_path,
                      RGBA="false",
                      filename_text_extension="true",
                  )

                  conditioningconcat_44 = conditioningconcat.concat(
                      conditioning_to=get_value_at_index(cliptextencode_45, 0),
                      conditioning_from=get_value_at_index(cliptextencode_4, 0),
                  )

                  imageresize_71 = imageresize.execute(
                      width=width,
                      height=height,
                      interpolation="nearest",
                      method="keep proportion",
                      condition="always",
                      multiple_of=0,
                      image=get_value_at_index(image_load_57, 0),
                  )

                  imageresize_69 = imageresize.execute(
                      width=width,
                      height=height,
                      interpolation="nearest",
                      method="keep proportion",
                      condition="always",
                      multiple_of=0,
                      image=get_value_at_index(image_load_58, 0),
                  )

                  maskfromrgbcmybw_65 = maskfromrgbcmybw.execute(
                      threshold_r=0.15,
                      threshold_g=0.15,
                      threshold_b=0.15,
                      image=get_value_at_index(imageresize_69, 0),
                  )

                  invertmask_62 = invertmask.invert(
                      mask=get_value_at_index(maskfromrgbcmybw_65, 6)
                  )

                  inpaintpreprocessor_37 = inpaintpreprocessor.preprocess(
                      image=get_value_at_index(imageresize_71, 0),
                      mask=get_value_at_index(invertmask_62, 0),
                  )

                  images_to_rgb_9 = images_to_rgb.image_to_rgb(
                      images=get_value_at_index(inpaintpreprocessor_37, 0)
                  )

                  controlnetapplyadvanced_10 = controlnetapplyadvanced.apply_controlnet(
                      strength=0.9,
                      start_percent=0,
                      end_percent=1,
                      positive=get_value_at_index(conditioningconcat_44, 0),
                      negative=get_value_at_index(cliptextencode_5, 0),
                      control_net=get_value_at_index(setunioncontrolnettype_30, 0),
                      image=get_value_at_index(images_to_rgb_9, 0),
                  )

                  samplercustom_73 = samplercustom.sample(
                      add_noise=True,
                      noise_seed=random.randint(1, 2**64),
                      cfg=1,
                      model=get_value_at_index(tcdmodelsamplingdiscrete_72, 0),
                      positive=get_value_at_index(controlnetapplyadvanced_10, 0),
                      negative=get_value_at_index(controlnetapplyadvanced_10, 1),
                      sampler=get_value_at_index(tcdmodelsamplingdiscrete_72, 1),
                      sigmas=get_value_at_index(tcdmodelsamplingdiscrete_72, 2),
                      latent_image=get_value_at_index(emptylatentimage_24, 0),
                  )

                  vaedecode_6 = vaedecode.decode(
                      samples=get_value_at_index(samplercustom_73, 0),
                      vae=get_value_at_index(vaeloader_27, 0),
                  )

                  image_save_29 = image_save.was_save_images(
                      output_path=output_path,
                      filename_prefix=f"{clean(textwrap.shorten(prompts[0], width=180))}-{num}-{name}-{ckpt_name}",
                      filename_delimiter="_",
                      filename_number_padding=4,
                      filename_number_start="false",
                      extension="webp",
                      dpi=96,
                      quality=80,
                      optimize_image="false",
                      lossless_webp="false",
                      overwrite_mode="false",
                      show_history="true",
                      show_history_by_prefix="false",
                      embed_workflow="true",
                      show_previews="false",
                      images=get_value_at_index(vaedecode_6, 0),
                  )


if __name__ == "__main__":
    main()
