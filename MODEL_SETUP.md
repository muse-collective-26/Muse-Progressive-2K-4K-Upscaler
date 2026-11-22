# Models and installation — Muse Progressive 2K / 4K

**Default recipe: standard Regenerate, Detailed 0.80. No IC LoRA or HyperFlow required.**

Download these files and place them in the following folders inside your ComfyUI installation:

| Model filename | Destination folder | Download |
| --- | --- | --- |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` | [Hugging Face — H3 diffusion models](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `ComfyUI/models/text_encoders/` | [Hugging Face — H3 text encoders](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| `minimax_h3_video_vae_fp16.safetensors` | `ComfyUI/models/vae/` | [Hugging Face — H3 VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae/` | [Hugging Face — H3 VAEs](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `open_h3_regenerate_2k_lora.safetensors` | `ComfyUI/models/loras/` | [Hugging Face — standard Regenerate LoRA](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/main/lora) |

The audio VAE is required for H3's silent internal audio latent. Your original audio bypasses generation and goes directly to the save nodes. No Sharp-curve or latent-upscaler checkpoint is needed.

## Required custom nodes

Install these under `ComfyUI/custom_nodes/` and follow each repository's requirements instructions:

- [Muse Progressive 2K 4K Upscaler](https://github.com/muse-collective-26/Muse-Progressive-2K-4K-Upscaler)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-Open-H3-Regenerate](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

Use current [ComfyUI](https://github.com/Comfy-Org/ComfyUI) with native MiniMax H3 and Comfy Kitchen/SOL attention. Install FFmpeg for video loading/export. Restart after installing nodes or models. Cloning the Muse repository alone does not download model weights or install the other node packages.

## Use

Choose a video, then select **2K only / 4K only / Both**. Detailed **0.80** is the default; **0.60** and **0.35** are softer options. Both stages use the selected strength. Keep the loader FPS and both save-node frame rates the same. Whole clips are sectioned automatically; set loader `frame_load_cap=22` only when you want a short test.

Long outputs use temporary disk-backed frames and can require substantial free disk space. Exact dimensions preserve the source shape with 32-pixel alignment; tested example: **2528×1440 → 3840×2208**. Read the repository README and validation record for hardware requirements and test coverage. Model licences remain separate from the node's code licence.
