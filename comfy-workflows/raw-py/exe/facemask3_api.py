import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

yolo_name="bbox/face_yolov8m.pt"
sam_name="sam_vit_b_01ec64.pth"
device_mode="Prefer GPU"
path="/content/drive/MyDrive/89_closeup/img"
count = 183
detection_hint_1="center-1"
detection_hint_2="diamond-4"
output_path="/content/drive/MyDrive/89_closeup/mask"

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


from nodes import ImageInvert, NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        ultralyticsdetectorprovider = NODE_CLASS_MAPPINGS[
            "UltralyticsDetectorProvider"
        ]()
        ultralyticsdetectorprovider_3 = ultralyticsdetectorprovider.doit(
            model_name=yolo_name
        )

        samloader = NODE_CLASS_MAPPINGS["SAMLoader"]()
        samloader_14 = samloader.load_model(
            model_name=sam_name, device_mode=device_mode
        )

        load_image_batch = NODE_CLASS_MAPPINGS["Load Image Batch"]()
        bboxdetectorsegs = NODE_CLASS_MAPPINGS["BboxDetectorSEGS"]()
        samdetectorcombined = NODE_CLASS_MAPPINGS["SAMDetectorCombined"]()
        masks_add = NODE_CLASS_MAPPINGS["Masks Add"]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        imageinvert = ImageInvert()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for q in range(count):
            load_image_batch_80 = load_image_batch.load_batch_images(
                mode="incremental_image",
                index=0,
                label="Batch 001",
                path=path,
                pattern="*",
                allow_RGBA_output="false",
                filename_text_extension="true",
            )

            bboxdetectorsegs_13 = bboxdetectorsegs.doit(
                threshold=0.5,
                dilation=10,
                crop_factor=3,
                drop_size=10,
                bbox_detector=get_value_at_index(ultralyticsdetectorprovider_3, 0),
                image=get_value_at_index(load_image_batch_80, 0),
            )

            samdetectorcombined_15 = samdetectorcombined.doit(
                detection_hint=detection_hint_1,
                dilation=0,
                threshold=0.1,
                bbox_expansion=0,
                mask_hint_threshold=1,
                mask_hint_use_negative="False",
                sam_model=get_value_at_index(samloader_14, 0),
                segs=get_value_at_index(bboxdetectorsegs_13, 0),
                image=get_value_at_index(load_image_batch_80, 0),
            )

            samdetectorcombined_84 = samdetectorcombined.doit(
                detection_hint=detection_hint_2,
                dilation=0,
                threshold=0.1,
                bbox_expansion=0,
                mask_hint_threshold=1,
                mask_hint_use_negative="False",
                sam_model=get_value_at_index(samloader_14, 0),
                segs=get_value_at_index(bboxdetectorsegs_13, 0),
                image=get_value_at_index(load_image_batch_80, 0),
            )

            masks_add_82 = masks_add.add_masks(
                masks_a=get_value_at_index(samdetectorcombined_15, 0),
                masks_b=get_value_at_index(samdetectorcombined_84, 0),
            )

            masktoimage_16 = masktoimage.mask_to_image(
                mask=get_value_at_index(masks_add_82, 0)
            )

            imageinvert_81 = imageinvert.invert(
                image=get_value_at_index(masktoimage_16, 0)
            )

            image_save_75 = image_save.was_save_images(
                output_path=output_path,
                filename_prefix=get_value_at_index(load_image_batch_80, 1),
                filename_delimiter="_",
                filename_number_padding=2,
                filename_number_start="false",
                extension="jpeg",
                quality=40,
                lossless_webp="false",
                overwrite_mode="false",
                show_history="false",
                show_history_by_prefix="true",
                embed_workflow="true",
                show_previews="true",
                images=get_value_at_index(imageinvert_81, 0),
            )


if __name__ == "__main__":
    main()
