import json,logging,time
from pathlib import Path
import torch
import comfy.model_management as mm
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import folder_paths,nodes,node_helpers
from .tile_runtime import FixedTileNoise,crop_conditioning
from .tile_layout import quarters,blend_mask
LOG=logging.getLogger(__name__)
SIGMAS=(0.8,0.49411764705882355,0.)
def call(name, **inputs):
    cls=nodes.NODE_CLASS_MAPPINGS[name]
    if hasattr(cls,'execute'):
        return cls.execute(**inputs)
    return getattr(cls(),cls.FUNCTION)(**inputs)

def _decode_parts(vae,samples):
    """[2026-11-23] How many pieces ComfyUI's H3 video VAE decodes this latent in (it tiles itself: 256 px tiles, one
    17-frame clip at a time). None when this ComfyUI's VAE doesn't expose that (then no per-piece progress)."""
    try:
        model=vae.first_stage_model
        latent=samples['samples']
        if getattr(latent,'is_nested',False):
            latent=latent.unbind()[0]
        if latent.ndim!=5 or not hasattr(model,'_decode_temporal_chunks') or not hasattr(model,'_adaptive_decode'):
            return None
        return int(model._decode_temporal_chunks(latent.shape[2])[1]) if latent.shape[2]>1 else 1
    except Exception:
        return None


def decode(samples,vae,notify=None,label='2K'):
    """Decode the refined latent to pixels.
    [2026-11-23] Rob (16 GB card): the decode printed nothing for 15+ minutes and looked frozen. It now reports each piece
    (console + node + progress bar) and its time. The tile settings below are ignored by the H3 video VAE (it always
    tiles itself, the lightest it can be), so the decode itself is unchanged."""
    notify=notify or (lambda *args:None)
    mm.unload_all_models()
    mm.soft_empty_cache()
    began=time.perf_counter()
    parts=_decode_parts(vae,samples)
    model=getattr(vae,'first_stage_model',None)
    LOG.warning('[MUSE UPSCALER] Decoding %s: %s (no console bar for this step).',label,
                f'{parts} piece(s)' if parts else 'one pass')
    notify(f'Decoding {label}'+(f' - piece 1/{parts}' if parts else ''))
    original=None
    if parts:
        import comfy.utils
        progress=comfy.utils.ProgressBar(parts)
        original=model._adaptive_decode
        done=[0]
        def counted(*args,**kwargs):
            out=original(*args,**kwargs)
            done[0]+=1
            took=time.perf_counter()-began
            LOG.warning('[MUSE UPSCALER] Decoding %s: piece %s/%s done (%.0f s so far)',label,done[0],parts,took)
            progress.update_absolute(min(done[0],parts))
            if done[0]<parts:
                notify(f'Decoding {label} - piece {done[0]+1}/{parts} ({took:.0f} s so far)')
            return out
        model._adaptive_decode=counted
    try:
        frames=call('VAEDecodeTiled',samples=samples,vae=vae,tile_size=512,overlap=64,temporal_size=64,temporal_overlap=8)[0]
    finally:
        if original is not None:
            try:
                del model._adaptive_decode   # back to the class method
            except AttributeError:
                model._adaptive_decode=original
    LOG.warning('[MUSE UPSCALER] Decoding %s finished in %.0f s',label,time.perf_counter()-began)
    return frames

class Detail4K:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'stage':('MUSE_LAYERED_STAGE',),
                            'detail_amount':('FLOAT',{'default':0.65,'min':0.,'max':1.5,'step':0.05,
                                                     'tooltip':'Experimental detail-layer contribution. 0 preserves the learned 4K enlargement; it does not add a detail layer.'}),
                            'overlap_pixels':('INT',{'default':128,'min':128,'max':256,'step':64})}}
    RETURN_TYPES=('IMAGE','STRING')
    RETURN_NAMES=('4K_video','tile_timings')
    FUNCTION='run'
    CATEGORY='Muse/H3 Layered Upscaler'
    DESCRIPTION='Progressive latent enlargement, sequential overlapping quarters, and an experimental band-limited correction layer. Standard Regenerate restoration; no IC/HyperFlow. Broad correction and finest residual noise are suppressed in latent coordinates. Quality must be checked.'

    def run(self,stage,detail_amount,overlap_pixels):
        return self.refine(stage,detail_amount,overlap_pixels)

    def refine(self,stage,detail_amount,overlap_pixels,light=False):
        start=time.perf_counter()
        notify=stage.get('progress_callback',lambda *args:None)
        notify('Preparing 4K latent and reference')
        models=stage['models']
        fw,fh=stage['final_size']
        enlarged=(call('VAEEncode',pixels=resize_pixels(stage['reference_frames'],fw,fh),vae=models['vae'])[0]
                  if stage.get('clean_start') else (_ for _ in ()).throw(ValueError('Clean-start stage required')))
        av=call('OpenH3RegenRefineLatent',latent=enlarged,audio_mode='silence (reference)',audio_vae=models['audio_vae'])[0]
        video,audio=av['samples'].unbind()
        canvas=video.detach().to(device='cpu',copy=True)
        audio=audio.cpu()
        conditioning=stage['conditioning']
        if light:
            # [2026-11-21] Qwen sees the decoded 2K result; DiT gets the exact
            # restored 2K latent as clean reference, without a VAE re-encode.
            reference_video=stage['latent']['samples'].unbind()[0].cpu()
            conditioning=call('MiniMaxH3ReferenceToVideo',clip=models['clip'],vae=None,prompt=stage.get('preservation_prompt',''),width=fw,height=fh,
                              length=len(stage['reference_frames']),ref_image_size='match',
                              ref_videos={'ref_video_0':stage['reference_frames']})[0]
            block={'kind':'video','latent_t':reference_video.shape[2],'latent_h':reference_video.shape[3],
                   'latent_w':reference_video.shape[4],'ref_audio_t':0,'latent':reference_video,'audio_latent':None}
            conditioning=node_helpers.conditioning_set_values(conditioning,{'minimax_refs':[block]})
            LOG.warning('[LIGHT 4K] Clean reference is restored 2K latent %sx%s; no reference VAE re-encode.',
                        reference_video.shape[-1]*16,reference_video.shape[-2]*16)
        schedule=((0.35,0.18,0.) if stage.get('clean_start') else (0.7,0.49411764705882355,0.)) if light else SIGMAS
        schedule=tuple(stage.get('refine_sigmas',schedule))
        tag='LIGHT 4K' if light else 'LAYERED 4K'
        height,width=canvas.shape[-2:]
        boxes=quarters(width,height,overlap_pixels//32)
        seed=stage['seed']
        noise_video,noise_audio=call('RandomNoise',noise_seed=seed)[0].generate_noise({'samples':comfy.nested_tensor.NestedTensor((canvas,audio))}).unbind()
        reports=[]
        progress=comfy.utils.ProgressBar(4)
        for i,box in enumerate(boxes):
            mm.throw_exception_if_processing_interrupted()
            notify(f'Refining 4K tile {i+1}/4')
            began=time.perf_counter()
            x0,y0,x1,y1=box
            alpha=blend_mask(box,i,boxes)
            before=canvas[...,y0:y1,x0:x1].clone()
            tile={'samples':comfy.nested_tensor.NestedTensor((before,audio)),
                  'noise_mask':comfy.nested_tensor.NestedTensor(((alpha>0).to(before.dtype).expand_as(before).contiguous(),torch.zeros_like(audio)))}
            cond=crop_conditioning(conditioning,box,width,height)
            LOG.warning('[%s TILE %s/4] START %sx%s, %s Regenerate steps, %s.',tag,i+1,(x1-x0)*16,(y1-y0)*16,
                        len(schedule)-1,'direct result, sharp 2K reference' if light else f'detail layer {detail_amount:.2f}')
            guider=call('BasicGuider',model=models['model'],conditioning=cond)[0]
            sampled=call('SamplerCustomAdvanced',noise=FixedTileNoise(seed,noise_video[...,y0:y1,x0:x1].contiguous(),noise_audio),
                         guider=guider,sampler=comfy.samplers.sampler_object('euler'),sigmas=torch.tensor(schedule),latent_image=tile)[0]
            refined=sampled['samples'].unbind()[0].cpu()
            if not bool(torch.isfinite(refined).all()):raise ValueError('Non-finite tile; refusing to save a fallback result.')
            if light:
                layered=refined
                delta=refined-before
                band=delta
            else:
                raise ValueError("Only direct two-step refinement is supported in this package")
            canvas[...,y0:y1,x0:x1]=before*(1-alpha)+layered*alpha
            elapsed=time.perf_counter()-began
            reports.append({'tile':i+1,'seconds':round(elapsed,3),'full_correction_rms':float(delta.square().mean().sqrt()),
                            'applied_layer_rms':float((detail_amount*band).square().mean().sqrt())})
            LOG.warning('[%s TILE %s/4] DONE %.1fs',tag,i+1,elapsed)
            progress.update_absolute(i+1)
            del sampled,refined,layered,delta,band,guider,cond,tile,before
        frames=decode({'samples':canvas},models['vae'],notify,'4K')   # [2026-11-23] progress per piece
        report={'size':[fw,fh],'frames':len(frames),'detail_amount':None if light else detail_amount,'overlap_pixels':overlap_pixels,
                'tiles':reports,'seconds_including_enlarge_decode':round(time.perf_counter()-start,3),
                'sigmas':schedule,'reference':'restored 2K latent' if light else 'original source latent',
                'method':('Sequential latent quarters; direct two-step Regenerate output with feathered fixed-context joins; no latent band filter.' if light else
                          'Sequential latent quarters, shared noise, inherited context; standard Regenerate/Euler restoration; 3x3 minus 7x7 spatially averaged correction band, scaled by detail amount. No temporal averaging. Experimental latent-frequency proxy, not guaranteed perceptual detail separation.')}
        destination=Path(folder_paths.get_output_directory())/'video'
        destination.mkdir(exist_ok=True)
        prefix='h3-light-refine' if light else 'h3-layered-detail'
        (destination/f'{prefix}-{time.time_ns()}.json').write_text(json.dumps(report,indent=2))
        return frames,json.dumps(report,indent=2)

class Light4K(Detail4K):
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'stage':('MUSE_LAYERED_STAGE',),
                            'overlap_pixels':('INT',{'default':128,'min':128,'max':256,'step':64})}}
    DESCRIPTION='Separate experiment: unchanged 2K base, sharp 2K latent reference, existing two-step Regenerate schedule, direct sampled output. No latent band filtering. Lower sigma does not guarantee unchanged faces or skin.'

    def run(self,stage,overlap_pixels):
        return self.refine(stage,1.,overlap_pixels,light=True)

def resize_pixels(images,width,height):
    return comfy.utils.common_upscale(images.movedim(-1,1),width,height,'lanczos','disabled').movedim(1,-1)


def colour_lock(frames, source, cell=32, batch=8):
    """[2026-11-22] Andy: the Regenerate pass drifted colour (darker, more magenta). Pull each frame's BROAD colour and
    brightness back to the source: blur both to ~cell-pixel blocks of the output, add the difference. Fine detail (anything
    smaller than a block) is untouched. frames/source: [N,H,W,C] 0..1, same frame count (source may be smaller)."""
    import torch.nn.functional as F
    n, h, w, _ = frames.shape
    lw, lh = max(8, round(w / cell)), max(8, round(h / cell))
    out = frames if frames.is_contiguous() else frames.contiguous()
    for i in range(0, min(n, len(source)), batch):
        f = out[i:i + batch].movedim(-1, 1).float()
        s = source[i:i + batch, ..., :3].to(f.device).movedim(-1, 1).float()
        low_f = F.interpolate(f, size=(lh, lw), mode='area')
        low_s = F.interpolate(s, size=(lh, lw), mode='area')
        diff = F.interpolate(low_s - low_f, size=(h, w), mode='bilinear', align_corners=False)
        out[i:i + batch] = (f + diff).clamp_(0, 1).movedim(1, -1).to(out.dtype)
    return out
