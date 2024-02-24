

import PIL
import os
import requests
import torch
from io import BytesIO
from datetime import datetime
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler
from accelerate import PartialState
from pathlib import Path

sd_model = "/content/ckpt/epiCRealism.safetensors.inpainting.safetensors" #@param {'type' : 'string'}



pipeline = StableDiffusionInpaintPipeline.from_single_file(
    sd_model,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

#@title ##  Anti Black
from diffusers.pipelines.stable_diffusion import safety_checker
def sc(self, clip_input, images) : return images, [False for i in images]
# edit the StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc


#@title ##  Lora loading

# pipeline.load_lora_weights("/content/Lora/", weight_name="meganekko-3.ckpt")
# pipeline.fuse_lora(lora_scale=0.8)
# pipeline.load_lora_weights("/content/Lora/", weight_name="dwrtsg-65-512-DB2.ckpt")
# pipeline.fuse_lora(lora_scale=0.5)
# pipeline.load_lora_weights("/content/Lora/", weight_name="add_detail.safetensors")
# pipeline.fuse_lora(lora_scale=1)
# pipeline.load_lora_weights("/content/Lora/", weight_name="lora-realistic-vision-51.safetensors")
# pipeline.fuse_lora(lora_scale=1)
pipeline.load_lora_weights("/content/Lora/", weight_name="lyco-epiCRealism.safetensors")
pipeline.fuse_lora(lora_scale=1)
# pipeline.load_lora_weights("/content/Lora/", weight_name="/content/Lora/edgCoquineHaute_Couture.safetensors")
# pipeline.fuse_lora(lora_scale=1)


#@title ##  Img Folders

img_mask_folder = (
  ('/content/Masking/content/Masking/B4u500/img','/content/Masking/content/Masking/B4u500/mask','jpeg'),
  ('/content/Masking/content/Masking/B4u550/img','/content/Masking/content/Masking/B4u550/mask','jpeg'),
  ('/content/Masking/content/Masking/B4u600/img','/content/Masking/content/Masking/B4u600/mask','jpeg'),
  ('/content/Masking/content/Masking/B4u650/img','/content/Masking/content/Masking/B4u650/mask','jpeg'),
  ('/content/Masking/content/Masking/B4u700/img','/content/Masking/content/Masking/B4u700/mask','jpeg'),
  ('/content/Masking/content/Masking/B4u750/img','/content/Masking/content/Masking/B4u750/mask','jpeg'),
  # ('/content/Masking/content/Masking/B4u800/img','/content/Masking/content/Masking/B4u800/mask','jpeg'),
  # ('/content/Masking/content/Masking/B4u850/img','/content/Masking/content/Masking/B4u850/mask','jpeg'),
  # ('/content/Masking/content/Masking/B3u500/img','/content/Masking/content/Masking/B3u500/mask','jpeg'),
  # ('/content/Masking/content/Masking/B3u550/img','/content/Masking/content/Masking/B3u550/mask','jpeg'),
  # ('/content/Masking/content/Masking/B3u600/img','/content/Masking/content/Masking/B3u600/mask','jpeg'),
  # ('/content/Masking/content/Masking/B3u650/img','/content/Masking/content/Masking/B3u650/mask','jpeg'),
  # ('/content/Masking/content/Masking/B3u700/img','/content/Masking/content/Masking/B3u700/mask','jpeg'),
  # ('/content/Masking/content/Masking/B3u750/img','/content/Masking/content/Masking/B3u750/mask','jpeg'),
  # ('/content/Masking/content/Masking/B3u850/img','/content/Masking/content/Masking/B3u850/mask','jpeg'),
  # ('/content/Masking/content/Masking/closeup/img','/content/Masking/content/Masking/closeup/mask','png'),
  ('/content/Masking/content/Masking/halfbody/img','/content/Masking/content/Masking/halfbody/mask','png'),
  ('/content/Masking/content/Masking/fullbody/img','/content/Masking/content/Masking/fullbody/img','png'),
)



dir = "/kaggle/working/outputs/"
timefolder = datetime.now().strftime("%Y%m%d%H")
from pathlib import Path
directory_path = Path(f"{dir}{timefolder}")
directory_path.mkdir(exist_ok=True)

wm_folder = "/content/colab/wildcard"
wm = WildcardManager(Path(wm_folder))
prrompt_template = "woman {||twintails} {blonde hair|}, wearing {||pink|red|green|Yellow|blue} { lolita|fairy|maid|princess|serafuku|ballgown|wedding|cute} dress, {|tutu|(tutu:0.5)}, glasses, {|kitchen|bed room|garden|cosmic dust|cyber punk city|white|simple} background, model photoshot, fashion photoshot, dynamic pose"  #@param {'type' : 'string'}
num_images = 1 # @param {type:"integer"}
iterate = 1 # @param {type:"integer"}

negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",

config_dict = {
  # "negative_prompt" : "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
  "num_inference_steps": 20,
  "width": 512,
  "height": 896,
  "num_images_per_prompt":1,
  "guidance_scale":6
}

for i in range(iterate):
  for img, mask, ext in img_mask_folder:

    image_folder = img
    mask_folder = mask
    mask_ext = ext

    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)

    for image_file in image_files:
      name, ext = os.path.splitext(image_file)
      if f'{name}.{mask_ext}' in mask_files:
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, name + '.' + mask_ext)

        image = PIL.Image.open(image_path)
        mask = PIL.Image.open(mask_path)

        resized_image = image.resize((512, 896))
        resized_mask = mask.resize((512, 896))

        generator = RandomPromptGenerator(wildcard_manager=wm)
        prompts = list(generator.generate(prrompt_template, num_images=num_images))

        for promptes in prompts:
          with distributed_state.split_between_processes([promptes, promptes]) as prompt:
              image = pipeline(**config_dict, prompt=prompt, image=resized_image, mask_image=resized_mask).images

              for img in image:
                now = str(datetime.now())
                img.save(f'{dir}{timefolder}/{now}.jpg')




