# Controlnet for Wan2.2

This repo contains the code for controlnet module for Wan2.2.  
Same approach as controlnet for [Wan2.1](https://github.com/TheDenk/wan2.1-dilated-controlnet).

### Models  
| Model | Processor | Huggingface Link |
|-------|:-----------:|:------------------:|
| TI2V-5B  | Depth     | [Link](https://huggingface.co/TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1)             |

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