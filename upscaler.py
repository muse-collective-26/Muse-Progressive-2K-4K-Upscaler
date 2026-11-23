"""Source-preserving H3 refinement; no Director or Studio dependency."""
import json
import logging
import math
import time
import tempfile
import shutil
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import comfy.samplers
import folder_paths
import nodes
from comfy_execution.graph import ExecutionBlocker
from .engine import call, decode, resize_pixels, Light4K, colour_lock

LOG = logging.getLogger(__name__)
PRESETS = {
    'Detailed · 0.80 (default)': (0.8, 0.49411764705882355, 0.),
    'Balanced · 0.60': (0.6, 0.36, 0.),
    'Soft · 0.35': (0.35, 0.18, 0.),
}
DEFAULT_PROMPT = 'Preserve the original subjects, identity, expressions, movement, lighting and colour. Refine existing hair, eyelashes, clothing and object detail. Keep skin texture consistent with the source; do not add freckles, new spots or uniform speckled pores.'


def output_sizes(width, height):
    """Match the tested 1440-short-edge base and 3840-long-edge final grids."""
    denominator = min(width, height) * 32
    base = tuple((v * 1440 + denominator - 1) // denominator * 32 for v in (width, height))
    denominator = max(base) * 32
    final = tuple((v * 3840 + denominator - 1) // denominator * 32 for v in base)
    return base, final


class MuseProgressiveUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        def model_list(category, default):
            return (folder_paths.get_filename_list(category), {'default': default})
        return {'hidden': {'unique_id': 'UNIQUE_ID'}, 'required': {
            'images': ('IMAGE',),
            'output': (['Both', '2K only', '4K only'], {'default': 'Both'}),
            'strength': (list(PRESETS), {'default': 'Balanced · 0.60'}),   # [2026-11-22] Andy: less face redraw by default
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
            'base_model': model_list('diffusion_models', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'),
            'text_encoder': model_list('text_encoders', 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
            'video_vae': model_list('vae', 'minimax_h3_video_vae_fp16.safetensors'),
            'audio_vae': model_list('vae', 'minimax_h3_audio_vae_fp32.safetensors'),
            'regenerate_lora': model_list('loras', 'open_h3_regenerate_2k_lora.safetensors'),
            'preservation_prompt': ('STRING', {'default': DEFAULT_PROMPT, 'multiline': True}),
            # [2026-11-22] Andy: appended at the end so saved workflows keep their values.
            'strength_4K': (['Same as 2K'] + list(PRESETS), {'default': 'Balanced · 0.60',
                'tooltip': 'How hard the 4K stage redraws the 2K result. 0.60 keeps good 4K detail with less face change; 0.80 = most detail, most change; 0.35 = gentle sharpen.'}),
            'colour_lock': ('BOOLEAN', {'default': True,
                'tooltip': 'After each stage, pull the overall colour and brightness back to the source video. Only broad colour is corrected; fine detail is kept.'}),
        }}

    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('video_2K', 'video_4K', 'report')
    FUNCTION = 'run'
    CATEGORY = 'Muse/Video Upscaling'
    DESCRIPTION = ('Standard Regenerate two-step refinement. Default sigma 0.8. '
                   '2K only skips 4K; 4K only builds the intermediate 2K internally. '
                   'Disabled outputs block their downstream save nodes. '
                   'Long clips are processed automatically in overlapping sections. '
                   'Long-clip outputs use disk-backed frame storage.')

    @staticmethod
    def _preview(unique_id, stage, frames=None):
        if unique_id is None:
            return
        try:
            from server import PromptServer
            event = {'node_id': str(unique_id), 'stage': stage}
            if frames is not None and len(frames):
                image = frames[len(frames)//2:len(frames)//2+1].detach().cpu()
                h, w = image.shape[1:3]
                scale = min(1., 768/max(h,w))
                image = resize_pixels(image, max(1,round(w*scale)), max(1,round(h*scale)))
                event['image'] = nodes.PreviewImage().save_images(image, filename_prefix='MuseProgressivePreview')['ui']['images'][0]
            PromptServer.instance.send_sync('muse_progressive_preview', event)
        except Exception:
            LOG.warning('[MUSE UPSCALER] Preview unavailable; render continues.', exc_info=True)

    def _models(self, base_model, text_encoder, video_vae, audio_vae, regenerate_lora):
        required = ['OpenH3RegenLora', 'OpenH3RegenRefineLatent', 'MiniMaxChunkFeedForward',
                    'ModelAttentionBackend', 'ModelPatchTorchSettings', 'BlockSparseAttention',
                    'MiniMaxH3ReferenceToVideo']
        missing = [n for n in required if n not in nodes.NODE_CLASS_MAPPINGS]
        if missing:
            raise RuntimeError('Missing ComfyUI nodes: ' + ', '.join(missing) + '. See this package README.')
        model = call('UNETLoader', unet_name=base_model, weight_dtype='default')[0]
        model = call('OpenH3RegenLora', model=model, lora_name=regenerate_lora, curve_name='none',
                     curve_table=base_model, strength=1., curve_strength=0., lora_dtype='bf16')[0]
        model = call('MiniMaxChunkFeedForward', model=model, chunks=2, seq_threshold=4096)[0]
        model = call('ModelAttentionBackend', model=model, attention='comfy kitchen attention')[0]
        model = call('ModelPatchTorchSettings', model=model, enable_fp16_accumulation=True)[0]
        model = call('BlockSparseAttention', model=model, selection={'selection': 'sol-attn', 'tau': 1.3},
                     start_percent=.2, end_percent=1., dense_blocks='', min_tokens=12288,
                     extra_tokens=256, sink_conditioning='exact_kv_and_rows', verbose=False)[0]
        return {'model': model,
                'clip': call('CLIPLoader', clip_name=text_encoder, type='minimax', device='default')[0],
                'vae': call('VAELoader', vae_name=video_vae)[0],
                'audio_vae': call('VAELoader', vae_name=audio_vae)[0]}

    def run(self, images, output, strength, seed, base_model, text_encoder, video_vae,
            audio_vae, regenerate_lora, preservation_prompt, strength_4K='Same as 2K', colour_lock=False, unique_id=None):
        notify = lambda stage, frames=None: self._preview(unique_id, stage, frames)
        notify("Starting - source preview", images)
        count = len(images)
        if count <= 124:
            return self._run_chunk(images, output, strength, seed, base_model, text_encoder,
                                   video_vae, audio_vae, regenerate_lora, preservation_prompt, notify=notify,
                                   strength_4K=strength_4K, lock=colour_lock)
        # Bound GPU work internally without imposing a user-facing clip-length cap.
        # Five inherited frames blend each section into its predecessor. Long output
        # buffers are file-backed, avoiding repeated full-length torch.cat copies.
        started = time.perf_counter()
        buffers, reports, storage = {}, [], []
        notify('Loading models')
        models = self._models(base_model, text_encoder, video_vae, audio_vae, regenerate_lora)
        start = 0
        while start < count:
            end = min(start + 124, count)
            result = self._run_chunk(images[start:end], output, strength, seed, base_model,
                text_encoder, video_vae, audio_vae, regenerate_lora, preservation_prompt, models=models, strength_4K=strength_4K, lock=colour_lock, notify=lambda stage, frames=None: notify(f"Frames {start+1}-{end}/{count}: {stage}", frames))
            reports.append(json.loads(result[2]))
            for slot in (0, 1):
                frames = result[slot]
                if isinstance(frames, ExecutionBlocker):
                    continue
                if slot not in buffers:
                    destination = Path(folder_paths.get_temp_directory()) / 'muse_progressive_frames'
                    destination.mkdir(parents=True, exist_ok=True)
                    required = count * math.prod(frames.shape[1:]) * 4
                    if shutil.disk_usage(destination).free < required + 256 * 1024**2:
                        raise RuntimeError(f'Insufficient free temporary disk space for this output: approximately {required / 1024**3:.1f} GiB required. Configure ComfyUI --temp-directory on a larger drive.')
                    handle, name = tempfile.mkstemp(prefix='frames-', suffix='.f32', dir=destination)
                    import os
                    os.close(handle)
                    mapped = np.memmap(name, mode='w+', dtype=np.float32,
                                       shape=(count, *frames.shape[1:]))
                    buffers[slot] = mapped
                    storage.append(name)
                mapped = buffers[slot]
                data = frames.detach().cpu().numpy()
                overlap = min(5, len(data)) if start else 0
                if overlap:
                    alpha = np.linspace(0, 1, overlap, dtype=np.float32)[:, None, None, None]
                    mapped[start:start+overlap] = mapped[start:start+overlap]*(1-alpha) + data[:overlap]*alpha
                mapped[start+overlap:end] = data[overlap:]
                mapped.flush()
            LOG.warning('[MUSE UPSCALER] Clip section complete: frames %s-%s of %s', start+1, end, count)
            if end == count:
                break
            start = end - 5
        values = [torch.from_numpy(buffers[k]) if k in buffers else ExecutionBlocker(None) for k in (0, 1)]
        report = {'frames': count, 'sections': reports, 'overlap_frames': 5,
                  'total_seconds': round(time.perf_counter()-started, 3),
                  'frame_storage': storage,
                  'storage_note': 'Temporary disk-backed outputs; retain until downstream saving completes. Use ComfyUI temporary-folder cleanup after the workflow finishes.'}
        return (*values, json.dumps(report, indent=2))

    def _run_chunk(self, images, output, strength, seed, base_model, text_encoder, video_vae,
                   audio_vae, regenerate_lora, preservation_prompt, models=None, notify=None,
                   strength_4K='Same as 2K', lock=False):
        started = time.perf_counter()
        count, height, width, channels = images.shape
        if count < 1:
            raise ValueError('Supply at least one video frame.')
        if min(width, height) < 128 or channels < 3:
            raise ValueError('Supply RGB video frames at least 128 pixels on each side.')
        sigmas = PRESETS[strength]
        if output not in ('Both', '2K only', '4K only'):
            raise ValueError('Unknown output selection')
        base_size, final_size = output_sizes(width, height)
        # Repeat-pad only the tail to the tested H3 sequence lengths, then trim outputs.
        padded_count = 5 if count <= 5 else 5 + math.ceil((count - 5) / 17) * 17
        source = images[..., :3].cpu()
        if padded_count > count:
            source = torch.cat((source, source[-1:].expand(padded_count-count, -1, -1, -1)), dim=0)
        # [2026-11-22] No edge padding: it was stretched into the 2K frame (912 wide -> padded 928 -> squeezed ~1.8% sideways,
        # with a smeared strip on the right/bottom). The VAE input is already resized to the 32-aligned 2K size, and the H3
        # reference node sizes its reference video itself.
        LOG.warning('[MUSE UPSCALER] Standard Regenerate; 2K %s, 4K %s, colour lock %s; %s frames; %s -> %s; output %s', strength, strength_4K, lock, count, base_size, final_size, output)
        notify = notify or (lambda *args: None)
        notify('Loading models / preparing 2K')
        models = models or self._models(base_model, text_encoder, video_vae, audio_vae, regenerate_lora)
        w, h = base_size
        latent = call('VAEEncode', pixels=resize_pixels(source, w, h), vae=models['vae'])[0]
        conditioning = call('MiniMaxH3ReferenceToVideo', clip=models['clip'], vae=models['vae'],
                            prompt=preservation_prompt, width=w, height=h, length=padded_count,
                            ref_image_size='match', ref_videos={'ref_video_0': source})[0]
        av = call('OpenH3RegenRefineLatent', latent=latent, audio_mode='silence (reference)', audio_vae=models['audio_vae'])[0]
        guider = call('BasicGuider', model=models['model'], conditioning=conditioning)[0]
        notify('Refining 2K - two steps')
        sampled = call('SamplerCustomAdvanced', noise=call('RandomNoise', noise_seed=seed)[0],
                       guider=guider, sampler=comfy.samplers.sampler_object('euler'),
                       sigmas=torch.tensor(sigmas), latent_image=av)[0]
        notify('Decoding 2K preview')
        frames = decode(sampled, models['vae'])
        if lock:   # [2026-11-22] colour back to the source (the 4K stage then starts from the corrected 2K too)
            frames = colour_lock(frames, source)
        notify('2K completed', frames)
        base_seconds = time.perf_counter() - started
        final = ExecutionBlocker(None)
        tile_report = None
        sigmas_4k = sigmas if strength_4K not in PRESETS else PRESETS[strength_4K]
        if output != '2K only':
            stage = {'latent': sampled, 'reference_frames': frames, 'conditioning': conditioning,
                     'models': models, 'seed': seed, 'final_size': final_size, 'clean_start': True,
                     'preservation_prompt': preservation_prompt, 'refine_sigmas': sigmas_4k, 'progress_callback': notify}
            final, tile_report = Light4K().run(stage, 128)
            if lock:
                final = colour_lock(final, source)
            final = final[:count]
            notify('4K completed', final)
        report = {'method': 'Standard Regenerate + sequential four-tile 4K', 'preset': strength,
                  'sigmas_2K': sigmas, 'sigmas_4K': sigmas_4k if output != '2K only' else None, 'colour_lock': bool(lock), 'seed': seed, 'frames': count, 'padded_frames': padded_count,
                  '2K_size': base_size, '4K_size': final_size if output != '2K only' else None,
                  '2K_seconds_including_setup': round(base_seconds, 3),
                  'total_seconds': round(time.perf_counter()-started, 3),
                  'tiles': json.loads(tile_report) if tile_report else None}
        LOG.warning('[MUSE UPSCALER] DONE %s', json.dumps(report))
        notify('Completed - ' + output)
        return (ExecutionBlocker(None) if output == '4K only' else frames[:count], final, json.dumps(report, indent=2))
