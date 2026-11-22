# Credits and scope

- MiniMax H3: MiniMax and the ComfyUI native integration contributors.
- Standard Open-H3-Regenerate-2K weights: NVIDIA / Efficient-Large-Model and the contributors credited by that model release.
- LoRA application and AV latent preparation are provided by the separately installed rdh9616/ComfyUI-Open-H3-Regenerate project (Apache-2.0). Its source is not bundled here.
- Attention controls, sampling and VAE operations are provided by ComfyUI; feed-forward chunking and Torch settings by KJNodes.
- Example video loading/export uses VideoHelperSuite.
- Muse code composes these components into source-preserving progressive refinement with sequential fixed-context overlapping spatial tiles and a single-node interface.

No model weights, reference videos or private Muse Director/Studio code are distributed in this repository. The MIT licence covers this package's code only. Dependencies and model weights retain their respective licences.
