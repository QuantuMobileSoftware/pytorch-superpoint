import cv2
import numpy as np
import torch

from .matching import Matching
from .superpoint.utils_numpy import filter_kp_by_masked_image

BRUTEFORCE_THRESHOLD = 1.05

class BruteForceMatching(Matching):

    def forward(self, data):
        """ Run SuperPoint and Bruteforce
        SuperPoint is skipped if ['pts_desc0', 'pts_desc1'] exist in input
        Args:
          data: dictionary with keys ['pts_desc0', 'pts_int0', 'image0', 'pts_desc1', 'pts_int1', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if torch.cuda.is_available() and "image0" in data:
            data["image0"] = data["image0"].to("cuda")
        if torch.cuda.is_available() and "image1" in data:
            data["image1"] = data["image1"].to("cuda")

        if 'pts_desc0' not in data:
            semi0, desc0 = self.superpoint( data['image0'])
            pred0 = self.superpoint.postprocess(semi0, desc0)
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'pts_desc1' not in data:
            semi1, desc1 = self.superpoint(data['image1'])
            pred1 = self.superpoint.postprocess(semi1, desc1)
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        data = {**data, **pred}

        data = {k: v.to("cpu").detach().numpy()[0] for k, v in data.items()}
        data["pts_int1"], data["pts_desc1"] = filter_kp_by_masked_image(data["image1"], data["pts_int1"], data["pts_desc1"])

        # no need to match, ransac requires at least 4 pts
        if data['pts_desc0'].shape[0] < 4 or data['pts_desc1'].shape[0] < 4:
            pred = {**data,
                    'matches0': np.zeros(data['pts_desc0'].shape[0], dtype=int) - 1,
                    'matches1': np.zeros(data['pts_desc1'].shape[0], dtype=int) - 1}
            return pred

        kpts0, kpts1 = data['pts_int0'], data['pts_int1']
        desc0, desc1 = np.ascontiguousarray(data['pts_desc0']), np.ascontiguousarray(data['pts_desc1'])

        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(desc0, desc1)

        new_matches0 = np.zeros((len(kpts0)), dtype=np.int32)-1
        new_matches1 = np.zeros((len(kpts1)), dtype=np.int32)-1
        for match in matches:
            if match.distance < BRUTEFORCE_THRESHOLD:
                new_matches0[match.queryIdx] = match.trainIdx
                new_matches1[match.trainIdx] = match.queryIdx

        data['matches0'] = new_matches0
        data['matches1'] = new_matches1
        return data
