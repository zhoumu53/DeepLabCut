# Code updated from ByteTrack
# https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/matching.py
# MIT License

import numpy as np
import torch

import scipy
# import lap
from scipy.spatial.distance import cdist

import deeplabcut.core.trackers.kalman_filter as kalman_filter

class DetectionResultsConverter:
    def __init__(self, bboxes, scores, poses, pose_scores, classes=None):
        
        self.tlwh = bboxes[:, :4]
        self.xywh = self.tlwh2xywh(self.tlwh)  # (n_individuals, 4) --- bboxes: tlwh -> xywh center: x,y - width, height        
        self.conf = scores  # (n_individuals, 1)
        self.poses = np.array(poses)[0]  ## poses: [1, n_keypoints, n_individuals, 2]
        ## transpose the poses to [n_individuals, n_keypoints, 2]
        self.poses = self.poses.transpose(1, 0, 2)
        self.pose_scores = np.array(pose_scores)  ## pose_scores: [n_individuals, n_keypoints, 1]

        self.cls = classes if classes is not None else np.zeros(len(scores))
        
    def tlwh2xywh(self, bboxes):
        """
        Convert bounding box format from [x, y, w, h] to [x, y, w, h] where x, y are the center coordinates.

        Args:
            bboxes (np.ndarray): Bounding box coordinates in tlwh format.

        Returns:
            np.ndarray: Bounding box coordinates in xywh format.
        """
        
        top_left = bboxes[:, :2]
        center_point = top_left + bboxes[:, 2:4]/2
        return np.concatenate([center_point, bboxes[:, 2:4]], axis=1)
        

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    Linear assignment using scipy.optimize.linear_sum_assignment
    
    Args:
        cost_matrix (np.ndarray): Cost matrix of shape (n_tracks, n_detections)
        thresh (float): Threshold for matching
        
    Returns:
        matches (np.ndarray): Array of shape (n_matches, 2) containing the indices of the matched tracks and detections
        unmatched_a (tuple): Tuple of indices of unmatched tracks
        unmatched_b (tuple): Tuple of indices of unmatched detections
    """
    
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    # Use scipy.optimize.linear_sum_assignment
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
    matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
    if len(matches) == 0:
        unmatched_a = list(np.arange(cost_matrix.shape[0]))
        unmatched_b = list(np.arange(cost_matrix.shape[1]))
    else:
        unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
        unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))


    return matches, unmatched_a, unmatched_b


def bbox_overlaps(box1, box2):
    """
    Compute IoU overlaps between two sets of bounding boxes.
    
    Parameters
    ----------
    box1: (N, 4) ndarray of float
    box2: (K, 4) ndarray of float
        
    Returns
    -------
    overlaps: (N, K) ndarray of IoU values
    """
    box1 = np.asarray(box1, dtype=np.float32)
    box2 = np.asarray(box2, dtype=np.float32)
    
    if box1.size == 0 or box2.size == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)
    
    # Unpack coordinates
    x1, y1, x2, y2 = box1.T
    qx1, qy1, qx2, qy2 = box2.T
    
    # Intersection (with broadcasting)
    ix1 = np.maximum(x1[:, None], qx1)
    iy1 = np.maximum(y1[:, None], qy1) 
    ix2 = np.minimum(x2[:, None], qx2)
    iy2 = np.minimum(y2[:, None], qy2)
    
    # Areas (inclusive coordinates like original Cython)
    inter = np.maximum(0, ix2 - ix1 + 1) * np.maximum(0, iy2 - iy1 + 1)
    area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area2 = (qx2 - qx1 + 1) * (qy2 - qy1 + 1)
    union = area1[:, None] + area2 - inter
    
    return inter / union


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    ious = bbox_overlaps(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def xywh2ltwh(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y