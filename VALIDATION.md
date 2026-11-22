# Validation record — 22 November 2026

## Selected recipe

Standard Open-H3-Regenerate LoRA, pruned H3, two Euler steps `0.8 → 0.49411765 → 0`, source-preserving pixel enlargement, sharp restored-2K latent reference for 4K, four sequential overlapping spatial tiles. No IC or HyperFlow in this release.

The preceding graph implementation completed the same 56-frame chunk at `2528 × 1440` and `3840 × 2208`. Total graph time was 421.49 seconds. The 4K stage including enlargement/reference preparation/decode reported 257.893 seconds; its four sampling tiles totalled 105.583 seconds. These categories are different and must not be presented as the same timing.

Native-size sampled frames showed more hair/eyelash/jewellery definition than the same 2K result simply resized to 4K. No blanket freckle pattern or obvious spatial join was identified in the inspected frames. This is a short-clip visual assessment, not proof of temporal stability on long videos or recovered ground-truth detail.

The IC + HyperFlow counterpart was also tested and retained separately: total 447.98 seconds, four sampling tiles 143.001 seconds. The methods differed in their native sampling schedules and reference preparation; these timings do not establish universal speed superiority. Andy selected the standard 0.8 output as clearer for this material.

## Automated checks

`tests/test_modes.py` passes five checks with heavy model operations mocked:

- 2K-only does not invoke 4K and returns a blocker for its disabled save branch.
- 4K-only still builds its intermediate, blocks the 2K output and trims final padding.
- All three presets propagate to both stages.
- A 244-frame input is split automatically into overlapping sections of 124, 124 and 6 frames; every original frame is retained and joins align correctly in the routing test.
- The tested source dimensions resolve exactly to `2528 × 1440 → 3840 × 2208`.

An integer alignment calculation replaced floating-point ceiling after this last check exposed a possible extra 32-pixel column. Mock checks do not establish image quality; see the GPU integration record below.

## Dependency revisions used during development

- ComfyUI: `c1eece62cbf1805d4f64286d7f5592e6a215e6ba`
- ComfyUI-Open-H3-Regenerate: `75124845aeff64e7dbad3b3ef3febaa03cc328ad`
- ComfyUI-KJNodes: `ab72eda7ffad6280422d71ab7d54d3fb87c0ba18`

Development installations may contain local changes; these hashes identify their checked-out revisions rather than a claim of pristine upstream reproducibility.

## GPU integration

The packaged node is tested separately from the earlier multi-node graph. Results are recorded in `integration-result.json` after completion. Only completed successful results should be treated as validation evidence.

Unvalidated: other GPU sizes/platforms, long-video visual continuity, a full quality comparison of the 0.6 preset, arbitrary model replacements and production delivery requirements.
