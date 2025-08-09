# Controlnet for Wan2.2

https://github.com/user-attachments/assets/36f6bd4f-450d-4aeb-afb9-02f62f3d6f34

This repo contains the code for controlnet module for Wan2.2.  
Same approach as controlnet for [Wan2.1](https://github.com/TheDenk/wan2.1-dilated-controlnet).  
```Currently, chess artifacts are observed in the 5B model inference. Perhaps this will be corrected in the future.```  

### For ComfyUI
Use the cool [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).  
<img width="600" height="480" alt="t2i_workflow" src="https://github.com/user-attachments/assets/4788c2d0-8ff0-405e-9b6d-0e0b1347865b" />

### Models  
| Model | Processor | Huggingface Link |
|-------|:-----------:|:------------------:|
| TI2V-5B  | Depth     | [Link](https://huggingface.co/TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1)             |
| TI2V-5B  | Canny     | [Link](https://huggingface.co/TheDenk/wan2.2-ti2v-5b-controlnet-canny-v1)             |
| TI2V-5B  | Hed     | [Link](https://huggingface.co/TheDenk/wan2.2-ti2v-5b-controlnet-hed-v1)             |
| T2V-A14B  | Depth     | [Link](https://huggingface.co/TheDenk/wan2.2-t2v-a14b-controlnet-depth-v1)             |
| T2V-A14B  | Hed     | [Link](https://huggingface.co/TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1)             |

### How to
Clone repo 
```bash
git clone https://github.com/TheDenk/wan2.2-controlnet.git
cd wan2.2-controlnet
```
  
Create venv  
```bash
python -m venv venv
source venv/bin/activate
```
  
Install requirements
```bash
pip install -r requirements.txt
```

### Inference examples
#### Simple inference with cli
```bash
python -m inference.cli_demo \
    --video_path "resources/bubble.mp4" \
    --prompt "Close-up shot with soft lighting, focusing sharply on the lower half of a young woman's face. Her lips are slightly parted as she blows an enormous bubblegum bubble. The bubble is semi-transparent, shimmering gently under the light, and surprisingly contains a miniature aquarium inside, where two orange-and-white goldfish slowly swim, their fins delicately fluttering as if in an aquatic universe. The background is a pure light blue color." \
    --controlnet_type "depth" \
    --base_model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --controlnet_model_path TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1
```

#### Detailed Inference
```bash
python -m inference.cli_demo \
    --video_path "resources/bubble.mp4" \
    --prompt "Close-up shot with soft lighting, focusing sharply on the lower half of a young woman's face. Her lips are slightly parted as she blows an enormous bubblegum bubble. The bubble is semi-transparent, shimmering gently under the light, and surprisingly contains a miniature aquarium inside, where two orange-and-white goldfish slowly swim, their fins delicately fluttering as if in an aquatic universe. The background is a pure light blue color." \
    --controlnet_type "depth" \
    --base_model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --controlnet_model_path TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1 \
    --controlnet_weight 0.8 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.8 \
    --controlnet_stride 3 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --video_height 480 \
    --video_width 832 \
    --num_frames 121 \
    --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 42 \
    --out_fps 24 \
    --output_path "result.mp4" \
    --teacache_treshold 0.6
```

#### Minimal code example
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffusers.utils import load_video, export_to_video
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from controlnet_aux import MidasDetector

from wan_controlnet import WanControlnet
from wan_transformer import CustomWanTransformer3DModel
from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline

base_model_path = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
controlnet_model_path = "TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1"
vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float32)
transformer = CustomWanTransformer3DModel.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
controlnet = WanControlnet.from_pretrained(controlnet_model_path, torch_dtype=torch.bfloat16)
pipe = WanTextToVideoControlnetPipeline.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    controlnet=controlnet,
    transformer=transformer,
    vae=vae, 
    torch_dtype=torch.bfloat16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
pipe.enable_model_cpu_offload()

controlnet_processor = MidasDetector.from_pretrained('lllyasviel/Annotators')
img_h = 704 # 704 480
img_w = 1280 # 1280 832
num_frames = 121  # 121 81 49

video_path = 'bubble.mp4'
video_frames = load_video(video_path)[:num_frames]
video_frames = [x.resize((img_w, img_h)) for x in video_frames]
controlnet_frames = [controlnet_processor(x) for x in video_frames]

prompt = "Close-up shot with soft lighting, focusing sharply on the lower half of a young woman's face. Her lips are slightly parted as she blows an enormous bubblegum bubble. The bubble is semi-transparent, shimmering gently under the light, and surprisingly contains a miniature aquarium inside, where two orange-and-white goldfish slowly swim, their fins delicately fluttering as if in an aquatic universe. The background is a pure light blue color."
negative_prompt = "bad quality, worst quality"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=img_h,
    width=img_w,
    num_frames=num_frames,
    guidance_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="pil",

    controlnet_frames=controlnet_frames,
    controlnet_guidance_start=0.0,
    controlnet_guidance_end=0.8,
    controlnet_weight=0.8,

    teacache_treshold=0.6,
).frames[0]

export_to_video(output, "output.mp4", fps=16)
```


## Acknowledgements
Original code and models [Wan2.2](https://github.com/Wan-Video/Wan2.2).  


## Citations
```
@misc{TheDenk,
    title={Wam2.2 Controlnet},
    author={Karachev Denis},
    url={https://github.com/TheDenk/wan2.2-controlnet},
    publisher={Github},
    year={2025}
}
```

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
