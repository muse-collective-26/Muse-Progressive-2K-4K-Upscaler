# 0.2.1

- The decode step after the two refine steps now shows its progress (console, node and progress bar): "Decoding 2K - piece
  3/7 (412 s so far)". Before, it printed nothing until it finished, so a slow decode on a 16 GB card looked frozen.
- The decode itself is unchanged: ComfyUI's H3 video VAE already decodes in its smallest pieces (256 px tiles, one
  17-frame clip at a time) whatever tile settings are passed.

# 0.2.0

- New **strength_4K** setting: the 4K stage has its own strength (default Balanced 0.60). Before, it always reused the
  2K strength, so the default 0.80 redrew faces and texture twice.
- New **colour_lock** setting (on by default): the broad colour and brightness of each stage's output are pulled back to
  the source; fine detail is unchanged.
- Default 2K strength is now **Balanced 0.60** (was Detailed 0.80). Saved workflows keep their own setting.
- The source is no longer padded to a multiple of 32 before resizing (it squeezed the picture ~1.8 % sideways and smeared
  a strip along the right/bottom edge). Sizes are still rounded to 32 px, which can stretch by up to ~1 %.
- New settings are added at the end of the node, so existing workflows keep their values.

# 0.1.0

- One node with 2K-only, 4K-only and Both output selection.
- Standard Regenerate 0.8 selected by default; optional 0.6 and 0.35 presets.
- Source-conditioned two-step 2K refinement followed by optional four-tile 4K refinement.
- No private Muse Director/Studio dependency and no model weights distributed.
- Whole-clip input with automatic overlapping temporal sections; no fixed input frame-count cap.
- Disk-backed long-output buffers and temporary-disk capacity checks.
- Simple four-node example workflow, model download guide and validation record.

Long-clip visual consistency and hardware compatibility beyond the development system remain beta.
