# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
from .superpoint.superpointnet import SuperPointNet
from .superpoint.utils import filter_kp_by_masked_image


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, superpoint_weights_path: str, superglue_weights_path: str = None, use_TTA: bool = False):
        """
        Args:
            superglue_weights_path (str): file with superglue weights (*.path)
        """
        super().__init__()
        self.superpoint = SuperPointNet(superpoint_weights_path).to(torch.device('cpu'))
        self.superpoint.eval()
        self.use_TTA = use_TTA

    def forward(self, data):
        if self.use_TTA:
            return self._forward_tta(data)
        else:
            return self._forward(data)

    def _forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
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
            semi0, desc0 = self.superpoint(data['image0'])
            pred0 = self.superpoint.postprocess(semi0, desc0)
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'pts_desc1' not in data:
            semi1, desc1 = self.superpoint(data['image1'])
            pred1 = self.superpoint.postprocess(semi1, desc1)
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        image1 = data['image1']
        kp1 = pred['pts_int1']
        desc1 = pred['pts_desc1']
        kp1, desc1 = filter_kp_by_masked_image(image1, kp1, desc1)
        pred['pts_int1'] = kp1
        pred['pts_desc1'] = desc1

        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        if data['pts_desc0'].shape[1] < 4 or data['pts_desc1'].shape[1] < 4:  # no need to match, ransac requires at least 4 pts
            pred = {**data,
                    'matches0': torch.zeros(max(data['pts_desc0'].shape[1], 1), dtype=torch.int64) - 1,
                    'matches1': torch.zeros(max(data['pts_desc1'].shape[1], 1), dtype=torch.int64) - 1,
                    }
            return pred

        # Perform the matching
        if torch.cuda.is_available():
            data = {k: v.to("cuda") for k, v in data.items()}
        pred = {**data, **self.superglue(data)}

        pred = {k: v.to("cpu").detach().numpy()[0] for k, v in pred.items()}
        return pred

    @staticmethod
    def _count_neighbours(points, distance_threshold):
        pairwise_distances = torch.cdist(points, points, p=2.0)

        num_neighbours = []
        for i in range(len(points)):
            num_neighbours.append(torch.count_nonzero(pairwise_distances[i] <= distance_threshold).item())

        return torch.Tensor(num_neighbours).type(torch.int32)
