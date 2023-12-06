"""losses
# losses for heatmap residule
# use it if you're computing residual loss. 
# current disable residual loss

"""
import torch


def pts_to_bbox(points, patch_size):  
    """
    input: 
        points: (y, x)
    output:
        bbox: (x1, y1, x2, y2)
    """

    shift_l = (patch_size+1) / 2
    shift_r = patch_size - shift_l
    pts_l = points-shift_l
    pts_r = points+shift_r+1
    bbox = torch.stack((pts_l[:, 1], pts_l[:, 0], pts_r[:, 1], pts_r[:, 0]), dim=1)
    return bbox
    pass


def _roi_pool(pred_heatmap, rois, patch_size=8):
    from torchvision.ops import roi_pool
    patches = roi_pool(pred_heatmap, rois.float(), (patch_size, patch_size), spatial_scale=1.0)
    return patches
    pass


def extract_patches(label_idx, image, patch_size=7):
    """
    return:
        patches: tensor [N, 1, patch, patch]
    """
    rois = pts_to_bbox(label_idx[:,2:], patch_size).long()
    rois = torch.cat((label_idx[:,:1], rois), dim=1)
    patches = _roi_pool(image, rois, patch_size=patch_size)
    return patches


def soft_argmax_2d(patches, device, normalized_coordinates=True):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
    import torchgeometry as tgm
    m = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    if patches.shape[0] > 0:
        coords = m(patches)
    else:
        coords = torch.zeros([0, 1, 2], device=device)
    return coords


def do_log(patches):
    patches[patches < 0] = 1e-6
    patches_log = torch.log(patches)
    return patches_log
