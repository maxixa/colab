import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

ckpt_name="realVisionV51.safetensors"
positive="high quality:1.4, 1girl, upper body, cafe"
negative="low quality:1.4, nsfw, full body"
width=512
height=512
batch_count=2
seed=random.randint(1, 2**64)
output_path="/content/drive/MyDrive/xyplot"

loras = [
    ['buke','granade'],
    ['mercon','karet']
]

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


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        efficient_loader = NODE_CLASS_MAPPINGS["Efficient Loader"]()


        xy_input_seeds_batch = NODE_CLASS_MAPPINGS["XY Input: Seeds++ Batch"]()
        xy_input_seeds_batch_38 = xy_input_seeds_batch.xy_value(batch_count=batch_count)

        xy_input_lora_block_weight_inspire = NODE_CLASS_MAPPINGS[
            "XY Input: Lora Block Weight //Inspire"
        ]()


        xy_plot = NODE_CLASS_MAPPINGS["XY Plot"]()
        ksampler_efficient = NODE_CLASS_MAPPINGS["KSampler (Efficient)"]()
        image_save = NODE_CLASS_MAPPINGS["Image Save"]()

        for lora_name,positive in loras:
            efficient_loader_10 = efficient_loader.efficientloader(
                ckpt_name=ckpt_name,
                negative=negative,
                empty_latent_width=width,
                vae_name="Baked VAE",
                clip_skip=-1,
                lora_name="None",
                lora_model_strength=1,
                lora_clip_strength=1,
                positive=positive,
                empty_latent_height=height,
                batch_size=1,
            )

            xy_input_lora_block_weight_inspire_40 = xy_input_lora_block_weight_inspire.doit(
                category_filter="All",
                lora_name=lora_name,
                strength_model=1,
                strength_clip=1,
                inverse=False,
                seed=random.randint(1, 2**64),
                A=4,
                B=1,
                preset="@SD-FULL-TEST:17",
                block_vectors="B1:A,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nB2:0,A,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nB3:0,0,A,0,0,0,0,0,0,0,0,0,0,0,0,0,0\nB4:0,0,0,A,0,0,0,0,0,0,0,0,0,0,0,0,0\nB5:0,0,0,0,A,0,0,0,0,0,0,0,0,0,0,0,0\nB6:0,0,0,0,0,A,0,0,0,0,0,0,0,0,0,0,0\nB7:0,0,0,0,0,0,A,0,0,0,0,0,0,0,0,0,0\nB8:0,0,0,0,0,0,0,A,0,0,0,0,0,0,0,0,0\nB9:0,0,0,0,0,0,0,0,A,0,0,0,0,0,0,0,0\nB10:0,0,0,0,0,0,0,0,0,A,0,0,0,0,0,0,0\nB11:0,0,0,0,0,0,0,0,0,0,A,0,0,0,0,0,0\nB12:0,0,0,0,0,0,0,0,0,0,0,A,0,0,0,0,0\nB13:0,0,0,0,0,0,0,0,0,0,0,0,A,0,0,0,0\nB14:0,0,0,0,0,0,0,0,0,0,0,0,0,A,0,0,0\nB15:0,0,0,0,0,0,0,0,0,0,0,0,0,0,A,0,0\nB16:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,A,0\nB17:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,A\n",
                heatmap_palette="viridis",
                heatmap_alpha=0.8,
                heatmap_strength=1.5,
                xyplot_mode="Simple",
            )

            xy_plot_11 = xy_plot.XYplot(
                grid_spacing=0,
                XY_flip="False",
                Y_label_orientation="Horizontal",
                cache_models="True",
                ksampler_output_image="Plot",
                dependencies=get_value_at_index(efficient_loader_10, 6),
                X=get_value_at_index(xy_input_lora_block_weight_inspire_40, 0),
                Y=get_value_at_index(xy_input_seeds_batch_38, 0),
            )

            ksampler_efficient_12 = ksampler_efficient.sample(
                sampler_state="Script",
                seed=seed,
                steps=20,
                cfg=10,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                preview_method="none",
                vae_decode="true",
                model=get_value_at_index(efficient_loader_10, 0),
                positive=get_value_at_index(efficient_loader_10, 1),
                negative=get_value_at_index(efficient_loader_10, 2),
                latent_image=get_value_at_index(efficient_loader_10, 3),
                optional_vae=get_value_at_index(efficient_loader_10, 4),
                script=get_value_at_index(xy_plot_11, 0),
            )

            image_save_42 = image_save.was_save_images(
                output_path=output_path,
                filename_prefix=f'{lora_name}_{ckpt_name}',
                filename_delimiter="_",
                filename_number_padding=2,
                filename_number_start="false",
                extension="png",
                quality=60,
                lossless_webp="false",
                overwrite_mode="false",
                show_history="false",
                show_history_by_prefix="true",
                embed_workflow="true",
                show_previews="true",
                images=get_value_at_index(ksampler_efficient_12, 5),
            )


if __name__ == "__main__":
    main()
