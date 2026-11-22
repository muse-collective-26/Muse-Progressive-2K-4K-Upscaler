# Muse Progressive 2K 4K Upscaler

One ComfyUI node: video frames in, 2K and/or tiled 4K frames out.

**Default: Detailed 0.80, standard Open-H3-Regenerate, two Euler steps.** This is the branch Andy selected after comparing actual renders. This package does **not** use the IC LoRA or HyperFlow. It does not require Muse Director or Muse Studio.

## Installation

From your `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/muse-collective-26/Muse-Progressive-2K-4K-Upscaler.git
```

Cloning installs this node's source only. Install the four listed dependencies (ComfyUI itself plus the three custom-node packages), download the five required models, and restart ComfyUI. ComfyUI Manager can install KJNodes, VideoHelperSuite and Open-H3-Regenerate, or clone their linked repositories into `custom_nodes` and follow their requirements instructions. Use ComfyUI's Python environment for any pip commands. No weights are downloaded automatically.

## Quick start

1. Install the dependencies and models below.
2. Place this folder in `ComfyUI/custom_nodes/Muse-Progressive-2K-4K-Upscaler` and restart ComfyUI.
3. Open `example_workflows/Muse Progressive Upscaler.json`.
4. Choose a video in **Load Video**. The example loads the whole clip (`frame_load_cap = 0`). For a quick test, temporarily set a 22-frame cap in the loader.
5. Select **2K only**, **4K only**, or **Both**, then Queue.

The workflow is Load Video → Muse Upscaler → Save 2K / Save 4K. Original audio is wired from the loader directly to both video-save nodes. The upscaler processes frames, not audio. Set the loader's frame rate and both save frame rates to the same FPS. The example uses 24 fps.

In 2K-only mode the 4K stage is never executed. In 4K-only mode the 2K intermediate is still required internally, but its output and save branch are blocked. Disabled branches may retain an older preview in the UI; they do not represent a newly saved video.

## Strength

| Preset | Two-step sigma schedule, at both stages | Status |
|---|---|---|
| **Detailed 0.80 — default** | `0.8 → 0.49411765 → 0` | Selected after the 56-frame comparison and tiled 4K test |
| Balanced 0.60 | `0.6 → 0.36 → 0` | Intermediate option; no equivalent full quality comparison yet |
| Soft 0.35 | `0.35 → 0.18 → 0` | More restrained; may add little detail |

Higher strength permits more reconstruction and can change faces or texture. The preservation prompt asks for source appearance, expression, movement and skin texture to remain consistent; it is guidance, not a hard constraint. The advanced model selectors and editable prompt live inside the same node, with working filenames preselected.

## Resolution

Labels describe output classes, not a forced aspect ratio. The 2K base targets a **1440-pixel short edge**, rounded up to 32-pixel alignment. The 4K stage targets a **3840-pixel long edge**, with the other edge rounded up to 32. Portrait stays portrait.

Example tested: `1344 × 768 → 2528 × 1440 → 3840 × 2208`. The 4K result is approximately 8.48 MP. It is not an exact UHD `3840 × 2160` export; crop separately if a delivery specification requires that exact size. Small alignment changes may slightly alter aspect ratio. This beta is intended for lower-resolution source clips, not already-4K input.

## Required ComfyUI dependencies

- A current [ComfyUI](https://github.com/Comfy-Org/ComfyUI) with native MiniMax H3, `ModelAttentionBackend`, `BlockSparseAttention`, Comfy Kitchen/SOL attention, and the matching CUDA/PyTorch environment.
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): `MiniMaxChunkFeedForward` and `ModelPatchTorchSettings`.
- [ComfyUI-Open-H3-Regenerate](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate): `OpenH3RegenLora` and `OpenH3RegenRefineLatent`.
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for loading/saving video in the example. Other frame loaders/savers can be connected instead.
- FFmpeg for video loading/export.

No additional pip packages are installed by this node. Use the dependencies' own installation instructions. The node checks required node types and reports missing dependencies instead of silently choosing a slower attention recipe.

## Required models

Weights are **not bundled or automatically downloaded**. Review each model repository's own licence and terms. This node's code licence does not grant rights to the models or their outputs.

| File | ComfyUI folder | Download source |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models` | [Comfy-Org MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `models/text_encoders` | [Comfy-Org MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae` | [Comfy-Org MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae` | Same VAE folder; needed for H3's silent audio latent even though original audio bypasses refinement |
| `open_h3_regenerate_2k_lora.safetensors` | `models/loras` | [Efficient-Large-Model/Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/tree/main/lora) |

Use the **standard** LoRA above, not the IC variant. No HyperFlow, latent-upscaler checkpoint or Sharp-curve file is required by this package. Existing files in configured model search paths can be reused; no private absolute paths are embedded.

## How it works

1. Repeat-pad the final frames to the tested H3 sequence lengths, then trim output back to the input frame count.
2. Resize source pixels with Lanczos to the 2K grid and encode using H3 VAE. Lanczos supplies the starting image; it is **not** the detail-generation pass.
3. Run source-conditioned H3 + standard Regenerate LoRA, CFG 1, two Euler steps, using the chosen sigma preset. Decode the 2K output.
4. If requested, resize the restored 2K frames to 4K and encode. Use the restored 2K latent as the clean reference.
5. Refine four sequential overlapping latent quarters, with shared global noise, fixed context from completed neighbours, 128-pixel overlap and feathered joins. Decode with tiled H3 VAE.

There is no latent band filter, extra sharpening or face replacement. Whole clips are automatically processed in sections of up to 124 frames, with five-frame overlaps and blended joins. The section size bounds GPU work; it does not limit input duration. Long outputs are stored as disk-backed float frame buffers in the ComfyUI temp folder, then read by downstream video saving. Allow substantial free temp-disk space (about 102 MB per frame for the tested 4K float buffers, plus any 2K output), and clean temporary files after downstream saving finishes. The input loader can still use substantial system RAM. Long-clip visual continuity remains beta and needs testing. Model setup uses the pruned H3 base, standard Regenerate LoRA at strength 1 in bf16, Sharp curves off, feed-forward chunking, FP16 accumulation and SOL sparse attention (`tau 1.3`, start `0.2`).

## Hardware and testing limits

Development hardware: RTX 5090 32 GB, 64 GB system RAM. Smaller GPUs and other platforms are **not validated**. Model downloads require substantial disk space. More frames increase source RAM, temporary disk space and processing time; the internal section size bounds GPU sampling work.

The selected recipe was tested on a **56-frame, 24 fps** clip: `2528 × 1440` intermediate, `3840 × 2208` final. The complete original graph took about **7m01s**, of which about **4m18s** was the 4K stage including enlargement and decoding. These are short-clip measurements, not a promise for full videos or other computers. Model cache and loading affect times. Detailed validation is recorded in `VALIDATION.md`.

Start the test server with `--disable-comfy-compiler`: a prior native model-patcher abort was avoided with this flag. This is a tested workaround, not a proven root-cause diagnosis. Existing server attention flags can remain configured as usual.

## Sharing

Repository: https://github.com/muse-collective-26/Muse-Progressive-2K-4K-Upscaler. Share that URL and this README in Skool. This local package does not publish itself or upload source clips, models or credentials. The example workflow refers to a local demo filename; community users select their own video.

This is a community **beta**, not a newly trained super-resolution model or a claim of recovered ground-truth detail. Check identity, motion, skin texture, fine-pattern stability and tile joins on your own material.

## In-node preview

A preview pane below the controls shows the source, then completed 2K and 4K frames. A live status line reports model loading, refinement, tile number and decoding. It is a completed-frame preview with live stage progress, not a decoded image at every diffusion step. Preview failures do not stop the render.

