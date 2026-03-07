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




# Global dictionary to store the current index for each folder path
# This ensures the counter persists across queue executions (looping behavior)
_folder_counters = {}

class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "mode": (["increment", "random", "fixed_index"], {"default": "increment"}),
                "fixed_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                # Optional seed input if you want repeatable randomness per run
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, folder_path, mode, fixed_index, seed=0):
        # Handle relative paths (relative to ComfyUI root)
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(folder_paths.base_path, folder_path)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Get valid image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        files.sort()  # Sort alphabetically for consistent ordering

        if len(files) == 0:
            raise FileNotFoundError(f"No images found in folder: {folder_path}")

        # --- Logic to determine which image to load ---
        current_index = 0

        if mode == "fixed_index":
            # Use the provided fixed index (modulo length to avoid errors)
            current_index = fixed_index % len(files)
        
        elif mode == "random":
            # Random pick based on seed
            random.seed(seed)
            current_index = random.randint(0, len(files) - 1)
        
        elif mode == "increment":
            # Loop Logic: Check global state
            if folder_path not in _folder_counters:
                _folder_counters[folder_path] = 0
            
            current_index = _folder_counters[folder_path]
            
            # Update counter for next time
            next_index = current_index + 1
            if next_index >= len(files):
                next_index = 0 # Loop back to start
            
            _folder_counters[folder_path] = next_index

        # Select the file
        selected_file = files[current_index]
        full_path = os.path.join(folder_path, selected_file)

        # --- Image Loading Logic (Standard ComfyUI approach) ---
        img = Image.open(full_path)
        
        output_mask = None
        output_image = None

        if img.mode == 'RGBA':
            # Separate RGB and Alpha
            img_data = np.array(img).astype(np.float32) / 255.0
            output_image = torch.from_numpy(img_data[:, :, :3]).unsqueeze(0)
            output_mask = torch.from_numpy(img_data[:, :, 3]).unsqueeze(0) # Shape: [1, H, W]
        
        elif 'A' in img.mode:
            # Handle other alpha modes (e.g., LA, PA)
            img = img.convert('RGBA')
            img_data = np.array(img).astype(np.float32) / 255.0
            output_image = torch.from_numpy(img_data[:, :, :3]).unsqueeze(0)
            output_mask = torch.from_numpy(img_data[:, :, 3]).unsqueeze(0)
            
        else:
            # Standard RGB images
            img = img.convert('RGB')
            img_data = np.array(img).astype(np.float32) / 255.0
            output_image = torch.from_numpy(img_data).unsqueeze(0)
            # Create a white mask (fully opaque)
            output_mask = torch.ones((1, img_data.shape[0], img_data.shape[1]), dtype=torch.float32)

        return (output_image, output_mask, selected_file)

    @classmethod
    def IS_CHANGED(s, folder_path, mode, fixed_index, seed=0):
        # This ensures the node triggers execution every time the queue runs
        # regardless of whether inputs changed.
        return float("NaN")

# Node Mappings
NODE_CLASS_MAPPINGS = {
    "Load Images From Folder": LoadImagesFromFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Images From Folder": "Load Images From Folder"
          }
