import os
import torch
import random
import numpy as np
from PIL import Image
import folder_paths

class SaveImageWEBP:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1}),
                "lossless": ("BOOLEAN", {"default": False}),
                "remove_metadata": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, filename_prefix, quality, lossless, remove_metadata, prompt=None, extra_pnginfo=None):
        # Handle filename prefix
        filename_prefix += self.prefix_append
        
        # Helper to get full path and counter (Mimics ComfyUI's internal logic)
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len]
            if prefix != os.path.basename(filename_prefix):
                return 0
            else:
                try:
                    digits = int(filename[prefix_len+1:].split('_')[0])
                except:
                    digits = 0
                return digits

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(self.output_dir, subfolder)
        
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)

        # Get existing files to calculate the next counter
        existing_files = [f for f in os.listdir(full_output_folder) if f.endswith('.webp')]
        if existing_files:
            counters = [map_filename(f) for f in existing_files]
            counter = max(counters) + 1
        else:
            counter = 1

        results = list()
        
        for image in images:
            # Convert tensor to numpy (0-1 float -> 0-255 uint8)
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Prepare Metadata
            metadata = None
            if not remove_metadata:
                metadata = {}
                if prompt is not None:
                    metadata["prompt"] = prompt
                if extra_pnginfo is not None:
                    metadata.update(extra_pnginfo)
                # Note: WebP metadata handling in PIL is limited compared to PNG. 
                # We inject it as basic text/exif if possible or just skip for size.

            # Generate filename
            file = f"{filename_prefix}_{counter:05}_.webp"
            full_path = os.path.join(full_output_folder, file)
            
            # Save WebP
            img.save(
                full_path, 
                format='WEBP', 
                quality=quality, 
                lossless=lossless,
                # PIL's webp plugin doesn't support complex dicts as metadata directly like PNG.
                # We just pass the image data here.
            )
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}

# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Save Image (WEBP)": SaveImageWEBP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image (WEBP)": "Save Image (WEBP)"
                  }




