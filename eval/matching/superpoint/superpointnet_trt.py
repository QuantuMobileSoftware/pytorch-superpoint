import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from datetime import datetime

from .utils import box_nms, getPtsFromHeatmap
from .utils_numpy import flattenDetection, extract_patches, soft_argmax_2d, do_log

from settings import (
    SUPERPOINT_INPUT_SHAPE
)


class SuperPointNet:
    """ TensorRT definition of SuperPoint Network. """
    def __init__(self, model_path, patch_size=5, nms_dist=4, conf_thresh=0.015):
        self.patch_size = patch_size
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.i = 1

        with open(model_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger())
            engine = runtime.deserialize_cuda_engine(f.read())

        self.context = engine.create_execution_context()

        input_dtype = trt.nptype(trt.float32)
        output_dtype = trt.nptype(trt.float32)

        input_bindings = []
        output_bindings = []

        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            if engine.binding_is_input(binding):
                input_data = np.zeros(size, dtype=dtype)
                input_bindings.append({"host": input_data, "device": cuda.mem_alloc(input_data.nbytes)})
            else:
                output_data = np.zeros(size, dtype=dtype)
                output_bindings.append({"host": output_data, "device": cuda.mem_alloc(output_data.nbytes)})


        self.input_dtype = trt.nptype(trt.float32)
        self.output_dtype = trt.nptype(trt.float32)

        self.input_bindings = input_bindings
        self.output_bindings = output_bindings

    def forward(self, x: np.ndarray):
        input_shape = (1, 1, *SUPERPOINT_INPUT_SHAPE[::-1])
        output_size1 = (1, 65, 30, 40)
        output_size2 = (1, 256, 30, 40)
        x = x.reshape(input_shape).astype(np.float32)

        input_bindings = self.input_bindings
        output_bindings = self.output_bindings

        # Transfer input data to the GPU
        cuda.memcpy_htod(input_bindings[0]["device"], x)
        # Execute the model
        self.context.execute_async_v2(
            bindings=[binding["device"] for binding in input_bindings] + [binding["device"] for binding in output_bindings],
            stream_handle=self.stream.handle
        )

        # Transfer output data from the GPU
        cuda.memcpy_dtoh(input_bindings[0]["host"], input_bindings[0]["device"])
        cuda.memcpy_dtoh(output_bindings[0]["host"], output_bindings[0]["device"])
        cuda.memcpy_dtoh(output_bindings[1]["host"], output_bindings[1]["device"])

        # Get the output tensors
        semi = output_bindings[0]["host"].reshape(output_size1)
        desc = output_bindings[1]["host"].reshape(output_size2)
        return semi, desc

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def postprocess(self, semi, desc):
        dn = np.linalg.norm(desc, ord=2, axis=1)
        desc = desc / np.expand_dims(dn, axis=1)

        output = {'semi': semi, 'desc': desc}
        heatmap = flattenDetection(semi)
        # nms
        heatmap_nms_batch = heatmap_to_nms(heatmap, nms_dist=self.nms_dist,
                                           conf_thresh=self.conf_thresh)
        outs = pred_soft_argmax(heatmap_nms_batch, heatmap, patch_size=self.patch_size)
        residual = outs['pred']
        # extract points
        outs = batch_extract_features(desc, heatmap_nms_batch, residual)
        output.update(outs)
        return output


def pred_soft_argmax(labels_2D, heatmap, patch_size):
    """

    return:
        dict {'loss': mean of difference btw pred and res}
    """
    outs = {}
    label_idx = np.stack(labels_2D.nonzero()).T
    patches = extract_patches(label_idx, heatmap, patch_size=patch_size)
    patches_log = do_log(patches)
    dxdy = soft_argmax_2d(patches_log, normalized_coordinates=False)
    dxdy = dxdy.squeeze(1)
    dxdy = dxdy - patch_size // 2

    outs['pred'] = dxdy
    outs['patches'] = patches
    return outs


def sample_desc_from_points(coarse_desc, pts, cell_size=8):
    """
    inputs:
        coarse_desc: ndarray of shape (1, 256, Hc, Wc)
        pts: ndarray of shape (N, 2)
    return:
        desc: ndarray of shape (1, N, D)
    """

    def grid_sample(input, grid, padding_mode='zeros', align_corners=False):
        """
        Optimized implementation of torch.nn.functional.grid_sample for 4-D inputs.

        Args:
            input (ndarray): input of shape (N, C, H_in, W_in)
            grid (ndarray): flow-field of shape (N, H_out, W_out, 2)
            padding_mode (str): padding mode for outside grid values ('zeros' is supported)
            align_corners (bool): if True, consider the input pixels as squares

        Returns:
            output (ndarray): output ndarray
        """
        N, C, H_in, W_in = input.shape
        N, H_out, W_out, _ = grid.shape

        output = np.zeros((N, C, H_out, W_out))

        # Apply align_corners transformation
        if align_corners:
            grid = (grid + 1) * 0.5
            grid[..., 0] *= W_in - 1
            grid[..., 1] *= H_in - 1
        else:
            grid[..., 0] = (grid[..., 0] + 1) * 0.5 * (W_in - 1)
            grid[..., 1] = (grid[..., 1] + 1) * 0.5 * (H_in - 1)

        if padding_mode == 'zeros':
            # Handle out-of-bound grid values
            mask = (grid[..., 0] >= 0) & (grid[..., 0] <= W_in - 1) & (grid[..., 1] >= 0) & (grid[..., 1] <= H_in - 1)
            valid_grid = grid[mask]
            valid_indices = np.where(mask)

            # Interpolation indices
            x0 = np.floor(valid_grid[..., 0]).astype(int)
            y0 = np.floor(valid_grid[..., 1]).astype(int)
            x1 = x0 + 1
            y1 = y0 + 1

            # Interpolation weights
            wx0 = valid_grid[..., 0] - x0
            wy0 = valid_grid[..., 1] - y0
            wx1 = 1 - wx0
            wy1 = 1 - wy0

            # Perform bilinear interpolation
            for c in range(C):
                output[valid_indices[0], c, valid_indices[1], valid_indices[2]] = (
                    wx1 * wy1 * input[valid_indices[0], c, y0, x0] +
                    wx0 * wy1 * input[valid_indices[0], c, y0, x1] +
                    wx1 * wy0 * input[valid_indices[0], c, y1, x0] +
                    wx0 * wy0 * input[valid_indices[0], c, y1, x1]
                )

        return output

    # --- Process descriptor.
    samp_pts = pts.T
    H, W = coarse_desc.shape[2] * cell_size, coarse_desc.shape[3] * cell_size
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.ones((1, 1, D))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = samp_pts.T.copy()
        samp_pts = samp_pts.reshape(1, 1, -1, 2)
        samp_pts = samp_pts.astype(float)

        desc = grid_sample(coarse_desc, samp_pts)
        desc = desc.squeeze(2).squeeze(0).T.reshape(1, -1, D)
    return desc


def heatmap_to_nms(heatmap, nms_dist, conf_thresh):
    """
    return:
      heatmap_nms_batch: np [batch, 1, H, W]
    """

    heatmap_nms_batch = [heatmap_nms(h, nms_dist, conf_thresh) for h in heatmap]
    heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
    heatmap_nms_batch = heatmap_nms_batch[:, np.newaxis, ...]
    return heatmap_nms_batch


def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):
    """
    input:
        heatmap: np [(1), H, W]
    """
    heatmap = heatmap.squeeze()
    pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)

    semi_thd_nms_sample = np.zeros_like(heatmap)
    semi_thd_nms_sample[pts_nms[1, :].astype(int), pts_nms[0, :].astype(int)] = 1

    return semi_thd_nms_sample


def batch_extract_features(desc, heatmap_nms_batch, residual):
    """
    return: -- type: dict
      desc: ndarray of shape (1, 256, 30, 40)
      heatmap_nms_batch: ndarray of shape (1, 1, 240, 320)
      residual: ndarray of shape (N, 2)
    """
    batch_size = heatmap_nms_batch.shape[0]

    pts_int, pts_desc = [], []
    pts_idx = np.argwhere(heatmap_nms_batch != 0)
    for i in range(batch_size):
        mask_b = (pts_idx[:, 0] == i)
        pts_int_b = pts_idx[mask_b][:, 2:].astype(float)
        pts_int_b = pts_int_b[:, [1, 0]]
        res_b = residual[mask_b]
        pts_b = pts_int_b + res_b
        pts_desc_b = sample_desc_from_points(desc[i].reshape(1, *desc[i].shape), pts_b).squeeze(0)
        pts_int.append(pts_int_b)
        pts_desc.append(pts_desc_b)
    
    pts_int = np.stack(pts_int, axis=0)
    pts_desc = np.stack(pts_desc, axis=0)
    return {'pts_int': pts_int, 'pts_desc': pts_desc}
