"""
Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
python -m inference.cli_demo \
    --video_path "./resources/bubble.mp4" \
    --prompt "Close-up shot with soft lighting, focusing sharply on the lower half of a young woman's face. Her lips are slightly parted as she blows an enormous bubblegum bubble. The bubble is semi-transparent, shimmering gently under the light, and surprisingly contains a miniature aquarium inside, where two orange-and-white goldfish slowly swim, their fins delicately fluttering as if in an aquatic universe. The background is a pure light blue color." \
    --controlnet_type "depth" \
    --base_model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --controlnet_model_path TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1
```

Additional options are available to specify the guidance scale, number of inference steps, video generation type, and output paths.
"""


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append('..')
import argparse

import cv2
import torch
import numpy as np
from PIL import Image

from transformers import UMT5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import export_to_video, load_video
from controlnet_aux import HEDdetector, CannyDetector, MidasDetector

from wan_controlnet import WanControlnet
from wan_transformer import CustomWanTransformer3DModel
from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline



def apply_gaussian_blur(image, ksize=5, sigmaX=1.0):
    image_np = np.array(image)
    if ksize % 2 == 0:
        ksize += 1
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return Image.fromarray(blurred_image)

class TilePreprocessor:
    def __call__(self, image, target_h, target_w, ksize=5, downscale_coef=4):
        img = image.resize((target_w // downscale_coef, target_h // downscale_coef))
        img = apply_gaussian_blur(img, ksize=ksize, sigmaX=ksize // 2)
        return img.resize((target_w, target_h))

def init_controlnet_processor(controlnet_type):
    if controlnet_type in ['canny', 'tile']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators').to(device='cuda')


controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
    'depth': MidasDetector,
    'tile': TilePreprocessor
}

@torch.no_grad()
def generate_video(
    prompt: str,
    video_path: str,
    base_model_path: str,
    controlnet_model_path: str,
    controlnet_type: str,
    controlnet_weight: float = 0.8,
    controlnet_guidance_start: float = 0.0,
    controlnet_guidance_end: float = 0.8,
    controlnet_stride: int = 3,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    teacache_treshold: float = 0.0,
    video_height: int = 480,
    video_width: int = 832,
    num_frames: int = 121,
    negative_prompt: str = "bad quality, worst quality",
    seed: int = 42,
    out_fps: int = 16,
    lora_path: str = None,
    lora_rank: int = 128,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - video_path (str): The video for controlnet processing.
    - base_model_path (str): The path of the pre-trained model to be used.
    - controlnet_model_path (str): The path of the pre-trained conrolnet model to be used.
    - controlnet_type (str): Type of controlnet model (e.g. canny, hed).
    - controlnet_weight (float): Strenght of controlnet
    - controlnet_guidance_start (float): The stage when the controlnet starts to be applied
    - controlnet_guidance_end (float): The stage when the controlnet end to be applied
    - controlnet_stride (int): Stride for controlnet blocks
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - teacache_treshold (float): TeaCache value. Best from [0.3, 0.5, 0.7, 0.9].
    - video_height (int): Output video height.
    - video_width (int): Output video width.
    - num_frames (int): Output frames count.
    - seed (int): The seed for reproducibility.
    - out_fps (int): FPS of output video.
    """

    # 0. Load selected controlnet processor
    controlnet_processor = init_controlnet_processor(controlnet_type)

    # 1.  Load the pre-trained Wan2.2 models with the specified precision (bfloat16).
    tokenizer = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float32)
    transformer = CustomWanTransformer3DModel.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    controlnet = WanControlnet.from_pretrained(controlnet_model_path, torch_dtype=torch.bfloat16)
    
    pipe = WanTextToVideoControlnetPipeline(
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae, 
        controlnet=controlnet,
        scheduler=scheduler,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.enable_model_cpu_offload()

    video = load_video(video_path)[:num_frames]
    video = [x.resize((video_width, video_height)) for x in video]
    if controlnet_type in ["tile"]:
        controlnet_frames = [controlnet_processor(x, video_height, video_width, ksize=5, downscale_coef=4) for x in video]
    else:
        controlnet_frames = [controlnet_processor(x) for x in video]

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)
    
    # 4. Generate the video frames based on the prompt.
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=video_height,
        width=video_width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="pil",
    
        controlnet_frames=controlnet_frames,
        controlnet_guidance_start=controlnet_guidance_start,
        controlnet_guidance_end=controlnet_guidance_end,
        controlnet_weight=controlnet_weight,
        controlnet_stride=controlnet_stride,

        teacache_treshold=teacache_treshold,
    ).frames[0]
    export_to_video(output, output_path, fps=out_fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan2.1")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/wan2.1-t2v-1.3b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--controlnet_type", type=str, default='hed', help="Type of controlnet model (e.g. canny, hed)")
    parser.add_argument("--controlnet_weight", type=float, default=0.8, help="Strenght of controlnet")
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.8, help="The stage when the controlnet end to be applied")
    parser.add_argument("--controlnet_stride", type=int, default=3, help="Strenght of controlnet")
    
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--video_height", type=int, default=704, help="Output video height")
    parser.add_argument("--video_width", type=int, default=1280, help="Output video width")
    parser.add_argument("--num_frames", type=int, default=121, help="Output frames count")
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--out_fps", type=int, default=16, help="FPS of output video")
    parser.add_argument("--teacache_treshold", type=float, default=0.0, help="TeaCache value. Best from [0.3, 0.5, 0.7, 0.9]")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    args = parser.parse_args()
    
    generate_video(
        prompt=args.prompt,
        video_path=args.video_path,
        base_model_path=args.base_model_path,
        controlnet_model_path=args.controlnet_model_path,
        controlnet_type=args.controlnet_type,
        controlnet_weight=args.controlnet_weight,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        controlnet_stride=args.controlnet_stride,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        video_height=args.video_height,
        video_width=args.video_width,
        num_frames=args.num_frames,
        seed=args.seed,
        out_fps=args.out_fps,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        teacache_treshold=args.teacache_treshold,
    )