import comfy.nested_tensor
from .tile_layout import reference_box

class FixedTileNoise:
    def __init__(self, seed, video, audio):
        self.seed = seed
        self.video = video
        self.audio = audio

    def generate_noise(self, latent):
        return comfy.nested_tensor.NestedTensor((self.video, self.audio))

def crop_conditioning(conditioning, box, width, height):
    result = []
    for embedding, metadata in conditioning:
        info = metadata.copy()
        refs = []
        for block in info.get('minimax_refs', []):
            if block['kind'] != 'video':
                raise ValueError('Tiled IC test expects one clean video reference, without image/audio references.')
            block = block.copy()
            z = block['latent']
            x0,y0,x1,y1 = reference_box(box, width, height, z.shape[-1], z.shape[-2])
            block['latent'] = z[..., y0:y1, x0:x1].contiguous()
            block['latent_h'], block['latent_w'] = y1-y0, x1-x0
            refs.append(block)
        if not refs:
            raise ValueError('Connect the source-video IC conditioning, including its video VAE reference.')
        info['minimax_refs'] = refs
        result.append([embedding, info])
    return result