import math
import torch


def quarters(width, height, overlap):
    if width % 2 or height % 2 or overlap % 2:
        raise ValueError('H3 tile dimensions and overlap must align to two latent cells (32 pixels).')
    mx, my = math.ceil(width / 4) * 2, math.ceil(height / 4) * 2
    if overlap < 4 or overlap >= min(mx, my):
        raise ValueError('Overlap must fit inside a quarter and contain at least four latent cells.')
    return [(0, 0, mx + overlap, my + overlap),
            (mx - overlap, 0, width, my + overlap),
            (0, my - overlap, mx + overlap, height),
            (mx - overlap, my - overlap, width, height)]


def blend_mask(box, index, boxes):
    x0, y0, x1, y1 = box
    wx, wy = torch.ones(x1-x0), torch.ones(y1-y0)
    # Outer half of each inherited overlap is fixed context; inner half blends.
    for axis, active, length in [(wx, index in (1, 3), boxes[0][2]-x0),
                                 (wy, index in (2, 3), boxes[0][3]-y0)]:
        if active:
            anchor = length // 2
            axis[:anchor] = 0
            axis[anchor:length] = torch.linspace(0, 1, length-anchor)
    return (wy[:, None] * wx[None, :])[None, None, None]


def reference_box(box, width, height, ref_width, ref_height):
    x0,y0,x1,y1 = box
    left = int(x0 / width * ref_width) // 2 * 2
    top = int(y0 / height * ref_height) // 2 * 2
    right = min(ref_width, math.ceil(x1 / width * ref_width / 2) * 2)
    bottom = min(ref_height, math.ceil(y1 / height * ref_height / 2) * 2)
    return left, top, right, bottom
