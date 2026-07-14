# Derived from MIT Driverless / cv-core, licensed under Apache License 2.0.
# Source: https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra (RektNet/cross_ratio_loss.py)
#
# MODIFIED: the geometric (cross-ratio) loss is re-derived for the BRT 8-keypoint layout. The
# original hard-codes RektNet's 7-point layout (apex + three left/right pairs) by index; BRT has
# no apex and pairs every level, so the collinearity and parallelism terms are rebuilt from the
# chain structure instead of fixed indices.
#
# The Apache-2.0 license text is preserved at LICENSE.apache-2.0.

"""Cross-ratio loss re-derived for the BRT keypoint layout.

The original loss hard-codes RektNet's 7-point layout (an apex plus three left/right pairs):

        0            apex
      1   2          mid_L_top, mid_R_top
      3   4          mid_L_bot, mid_R_bot
      5   6          bot_L,     bot_R

and enforces two things that hold for any cone under perspective:
  * the left silhouette (0-1-3-5) and right silhouette (0-2-4-6) are each straight lines
  * the horizontals (1-2, 3-4, 5-6) -- the colour-band boundaries -- stay parallel

BRT has no apex and pairs every level, so the same two constraints apply to a shorter chain:

      0   1          top pair
      2   3          mid pair
      4   5          base pair
     (6   7)         extra pair, large cones only

Dropping the apex costs nothing geometrically: collinearity of a 3-point chain is exactly what
the original enforced between consecutive edges. Keeping this loss is the point -- it is what
makes RektNet robust, and comparing a RektNet stripped of it against YOLO-pose would be a
strawman.
"""
import torch
import torch.nn.functional as F
from torch import nn


def _collinear(a, b, c):
    """Penalty for a-b-c not lying on a straight line (0 when the edges are parallel)."""
    v1 = F.normalize(b - a, dim=1)
    v2 = F.normalize(c - b, dim=1)
    return 1.0 - (v1 * v2).sum(dim=1)


def _parallel(a, b, c, d):
    """Penalty for segment a-b not being parallel to segment c-d."""
    v1 = F.normalize(b - a, dim=1)
    v2 = F.normalize(d - c, dim=1)
    # No abs(): the original does not take one, and taking it would make a 180-degree flipped
    # horizontal (left/right pair swapped) a zero-loss configuration -- forgiving precisely the
    # failure mode the flip augmentation can introduce.
    return 1.0 - (v1 * v2).sum(dim=1)


class BRTCrossRatioLoss(nn.Module):
    """Location loss plus the cone-geometry priors, for the 4/6/8-keypoint BRT layouts."""

    def __init__(self, loss_type="l1_softargmax", include_geo=True,
                 geo_loss_gamma_horz=0.0, geo_loss_gamma_vert=0.0, num_kpt=6):
        super().__init__()
        self.loss_type = loss_type
        self.include_geo = include_geo
        self.gamma_horz = geo_loss_gamma_horz
        self.gamma_vert = geo_loss_gamma_vert
        self.num_kpt = num_kpt

        # Left indices are even, right are odd. Descending the cone the levels are
        # top (0,1) -> mid (2,3) -> extra (6,7) -> base (4,5): the extra pair sits *above* the
        # base, not below it, so the chain order is not simply the index order.
        if num_kpt == 8:
            self.left_chain = [0, 2, 6, 4]
            self.right_chain = [1, 3, 7, 5]
            self.horizontals = [(0, 1), (2, 3), (6, 7), (4, 5)]
        elif num_kpt == 6:
            self.left_chain = [0, 2, 4]
            self.right_chain = [1, 3, 5]
            self.horizontals = [(0, 1), (2, 3), (4, 5)]
        else:  # 4 keypoints: a top and a base pair, nothing in between to be collinear with
            self.left_chain = []
            self.right_chain = []
            self.horizontals = [(0, 1), (2, 3)]

    def forward(self, heatmap, points, target_hm, target_points, vis=None):
        """vis: (B, K) 1/0 mask. Points a cone does not have (kpt6/7 on a small cone) must not
        be supervised -- their target coordinates are padding, and training on them teaches the
        model to hallucinate a stripe boundary that is not there."""
        if vis is None:
            vis = torch.ones(points.shape[:2], device=points.device, dtype=points.dtype)
        m = vis.unsqueeze(-1)                       # (B, K, 1) for the coordinate terms
        denom = vis.sum().clamp(min=1.0)

        if self.loss_type in ("l2_softargmax", "l2_sm"):
            location_loss = (((points - target_points) ** 2) * m).sum() / denom
        elif self.loss_type in ("l2_heatmap", "l2_hm"):
            hm_mask = vis.view(*vis.shape, 1, 1)
            location_loss = (((heatmap - target_hm) ** 2) * hm_mask).sum() / denom
        elif self.loss_type in ("l1_softargmax", "l1_sm"):
            location_loss = ((points - target_points).abs() * m).sum() / denom
        else:
            raise ValueError(f"unknown loss type: {self.loss_type}")

        if not self.include_geo:
            return location_loss, torch.zeros((), device=points.device), location_loss

        def geo_mask(*idx):
            """A geometric term only applies where every point it involves actually exists."""
            out = vis[:, idx[0]]
            for i in idx[1:]:
                out = out * vis[:, i]
            return out

        def masked_mean(term, mask):
            return (term * mask).sum() / mask.sum().clamp(min=1.0)

        vert_terms = []
        for chain in (self.left_chain, self.right_chain):
            for i in range(len(chain) - 2):
                a, b, c = chain[i], chain[i + 1], chain[i + 2]
                vert_terms.append(
                    masked_mean(_collinear(points[:, a], points[:, b], points[:, c]),
                                geo_mask(a, b, c))
                )

        horz_terms = []
        for i in range(len(self.horizontals) - 1):
            (a, b), (c, d) = self.horizontals[i], self.horizontals[i + 1]
            horz_terms.append(
                masked_mean(_parallel(points[:, a], points[:, b], points[:, c], points[:, d]),
                            geo_mask(a, b, c, d))
            )

        vert = torch.stack(vert_terms).mean() if vert_terms else torch.zeros((), device=points.device)
        horz = torch.stack(horz_terms).mean() if horz_terms else torch.zeros((), device=points.device)
        geo_loss = self.gamma_vert * vert + self.gamma_horz * horz

        return location_loss, geo_loss, location_loss + geo_loss
