# Code updated from ByteTrack
# https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py
# MIT License

import numpy as np
import torch
import torch.nn.functional as F

from deeplabcut.core.trackers.basetrack import BaseTrack, TrackState
from deeplabcut.core.trackers.kalman_filter import KalmanFilter
from deeplabcut.core.trackers.utils import *

def xywh2tlwh(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in tlwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, pose=None, pose_score=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        
        self.last_pose = None if pose is None else np.asarray(pose, dtype=np.float32)  # (K,2)
        self.last_pose_score = None if pose_score is None else np.asarray(pose_score, dtype=np.float32)  # (K,1)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.last_pose = new_track.last_pose

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.last_pose = new_track.last_pose

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
        ## new: fixed-id tracking
        self.K = int(args.n_individuals)
        self.fixed_ids = list(range(1, self.K+1))
        self.available_ids = self.fixed_ids.copy()
        self.full_init = False

        self.revive_enable = getattr(args, "revive_enable", True)
        self.revive_iou_thresh = getattr(args, "revive_iou_thresh", 0.7)  # cost <= 0.7 -> IoU >= 0.3
        self.revive_maha_gate = getattr(args, "revive_maha_gate", 16.27)  # looser than 9.21
        self.revive_max_gap = getattr(args, "revive_max_gap", self.max_time_lost)
        self.revive_use_pose = getattr(args, "revive_use_pose", True)
        self.revive_pose_w = getattr(args, "revive_pose_w", 0.4)
        
        # new: init warmup frames
        self.init_warmup_frames = int(getattr(args, "init_warmup_frames", 3)) 
        self.init_rank_by = str(getattr(args, "init_rank_by", "x"))
        self.init_require_hits = int(getattr(args, "init_require_hits", 2))
        self._warmup_stats = {}

    def update(self, output_results, img=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        scores = output_results.conf
        bboxes = output_results.xywhr if hasattr(output_results, "xywhr") else output_results.xywh
        bboxes_tlwh = output_results.tlwh
        poses = output_results.poses if hasattr(output_results, "poses") else None  ## (n_individuals, n_keypoints, 2)
        pose_scores = output_results.pose_scores if hasattr(output_results, "pose_scores") else None  ## (n_individuals, n_keypoints, 1)

        # print(f"Pose: {poses} -=-=-=-")
        # print(f"Pose scores: {pose_scores.shape} -=-=-=-")
        
        mask_high = scores >= self.args.track_high_thresh
        mask_low  = scores >  self.args.track_low_thresh
        mask_mid  = np.logical_and(mask_low, ~mask_high)

        no_tracks_yet = (self.frame_id == 1) and (len(self.tracked_stracks) == 0) and (len(self.lost_stracks) == 0)
        keep_mask = np.logical_or(mask_high, mask_mid) if no_tracks_yet else mask_high

        dets = bboxes_tlwh[keep_mask]
        scores_keep = scores[keep_mask]
        poses_keep = poses[keep_mask] if poses is not None else None
        pose_scores_keep = pose_scores[keep_mask] if pose_scores is not None else None

        dets_second = bboxes_tlwh[mask_mid]
        scores_second = scores[mask_mid]
        poses_second = poses[mask_mid] if poses is not None else None
        pose_scores_second = pose_scores[mask_mid] if pose_scores is not None else None
        
        if len(dets) > 0:
            '''Detections'''
            # detections = [STrack(tlwh, s) for
            #               (tlwh, s) in zip(dets, scores_keep)]
            detections = [STrack(tlwh, s, pose=(poses_keep[i] if poses_keep is not None else None), pose_score=(pose_scores_keep[i] if pose_scores_keep is not None else None))
                for i, (tlwh, s) in enumerate(zip(dets, scores_keep))]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # print(f"Adding track {track.track_id} to unconfirmed -=-=-=-")
                unconfirmed.append(track)
            else:
                # print(f"Adding track {track.track_id} to tracked_stracks -=-=-=-")
                tracked_stracks.append(track)
                
        # print(f"Strack pool: {len(tracked_stracks)} -=-=-=-")
        # print(f"Lost stracks: {len(self.lost_stracks)} -=-=-=-")

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists_iou = iou_distance(strack_pool, detections)
        # print(f"Distances: {np.array(dists).shape} -=-=-=-")
        if self.args.fuse_score:
            dists_iou = fuse_score(dists_iou, detections)
        
        if self.revive_use_pose:
            n_bodyparts = detections[0].last_pose.shape[0]
            pose_sigmas = np.array([0.1]* n_bodyparts)
            dists_pose = _oks_cost_tracks_vs_dets(strack_pool, detections, sigmas=pose_sigmas)
            # print(f"Distances pose: {np.array(dists_pose).shape} -=-=-=-")
            dists = (1.0 - self.revive_pose_w) * dists_iou + self.revive_pose_w * dists_pose
        else:
            dists = dists_iou
            
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)  ## 
        # print("================================================step 2=================================================")
        # print(f"Matches: {matches} -=-=-=-")
        # print(f"Unmatched tracks: {u_track} -=-=-=-")
        # print(f"Unmatched detections: {u_detection} -=-=-=-")

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(tlwh, s, pose=(poses_second[i] if poses_second is not None else None), pose_score=(pose_scores_second[i] if pose_scores_second is not None else None))
                                for i, (tlwh, s) in enumerate(zip(dets_second, scores_second))]
            # detections_second = [STrack(tlwh, s) for
            #               (tlwh, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # print(f"R tracked stracks: {len(r_tracked_stracks)} -=-=-=-")
        dists_iou = iou_distance(r_tracked_stracks, detections_second)
        if self.args.fuse_score:
            dists_iou = fuse_score(dists_iou, detections_second)
        
        if self.revive_use_pose:
            dists_pose = _oks_cost_tracks_vs_dets(r_tracked_stracks, detections_second, sigmas=pose_sigmas)
            dists = (1.0 - self.revive_pose_w) * dists_iou + self.revive_pose_w * dists_pose
        else:
            dists = dists_iou
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        # print("================================================step 3=================================================")
        # print(f"Distances: {np.array(dists).shape} -=-=-=-")
        # print(f"Matches: {matches} -=-=-=-")
        # print(f"Unmatched tracks: {u_track} -=-=-=-")
        # print(f"Unmatched detections: {u_detection_second} -=-=-=-")
        for itracked, idet in matches:
            # print(f"Matching track {itracked} to detection {idet} -=-=-=-")
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        if self.args.fuse_score:
            dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # """ Step 4: Init new stracks"""
        # for inew in u_detection:
        #     track = detections[inew]
        #     if track.score < self.args.new_track_thresh:
        #         continue
        #     track.activate(self.kalman_filter, self.frame_id)
        #     activated_starcks.append(track)
        
        """ Step 4: Init new stracks (revive-first, no new IDs after full init) """
        if len(u_detection) > 0:
            cand_dets = [detections[i] for i in u_detection]

            # try to revive lost tracks first - reuse old IDs
            revived = []
            still_unmatched = list(range(len(cand_dets)))  # indices into cand_dets

            if self.revive_enable and len(self.lost_stracks) > 0:
                recent_lost = [t for t in self.lost_stracks
                            if (self.frame_id - t.end_frame) <= self.revive_max_gap]
                if len(recent_lost) > 0:
                    STrack.multi_predict(recent_lost)

                    gate = _maha_gate_tracks_vs_dets(recent_lost, cand_dets, chi2_thr=self.revive_maha_gate)
                    C_iou = _iou_cost_tracks_vs_dets(recent_lost, cand_dets)
                    C = np.where(gate, C_iou, 1e3)

                    if self.revive_use_pose:
                        C_oks = _oks_cost_tracks_vs_dets(recent_lost, cand_dets)
                        C = (1.0 - self.revive_pose_w) * C + self.revive_pose_w * C_oks

                    matches, u_lost, u_cand = linear_assignment(C, thresh=self.revive_iou_thresh)

                    for ilost, icand in matches:
                        lt = recent_lost[ilost]
                        det = cand_dets[icand]
                        lt.re_activate(det, self.frame_id, new_id=False)  # keep same ID
                        revived.append(lt)
                        if icand in still_unmatched:
                            still_unmatched.remove(icand)

                    if len(revived) > 0:
                        refind_stracks.extend(revived)
                        self.lost_stracks = sub_stracks(self.lost_stracks, revived)

            # IMPORTANT: Do NOT create brand-new IDs after full init.
            # Only fill remaining fixed IDs *before* full init is complete.
            if not self.full_init and len(self.available_ids) > 0 and len(still_unmatched) > 0:
                # sort remaining candidates by score desc
                # (find their original indices in scores_keep if needed; here we sort by det.score)
                still_unmatched.sort(key=lambda idx: float(cand_dets[idx].score), reverse=True)

                births = []
                while len(self.available_ids) > 0 and len(still_unmatched) > 0:
                    idx = still_unmatched.pop(0)
                    det = cand_dets[idx]
                    # initialize track then overwrite ID with a fixed one
                    det.activate(self.kalman_filter, self.frame_id)
                    det.track_id = self.available_ids.pop(0)
                    det.is_activated = True
                    births.append(det)
                    activated_starcks.append(det)

                # If we just assigned the last fixed ID, lock full_init and block all future births
                if len(self.available_ids) == 0:
                    self.full_init = True
                    self.args.new_track_thresh = float('inf')

        # print("======================= lost stracks =========================", len(self.lost_stracks))   
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                # print(f"Marking track {track.track_id} as removed")
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # print(f"Lost stracks: {self.lost_stracks} , {len(self.lost_stracks)} -=-=-=-")
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # print()
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def iou_single(a_tlbr: np.ndarray, b_tlbr: np.ndarray, eps: float = 1e-6) -> float:
    x1 = max(a_tlbr[0], b_tlbr[0])
    y1 = max(a_tlbr[1], b_tlbr[1])
    x2 = min(a_tlbr[2], b_tlbr[2])
    y2 = min(a_tlbr[3], b_tlbr[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    a_area = max(0.0, a_tlbr[2] - a_tlbr[0]) * max(0.0, a_tlbr[3] - a_tlbr[1])
    b_area = max(0.0, b_tlbr[2] - b_tlbr[0]) * max(0.0, b_tlbr[3] - b_tlbr[1])
    union = a_area + b_area - inter + eps
    return inter / union

# [REVIVE] Gating against detections (list[STrack]) using track means/covs
def _maha_gate_tracks_vs_dets(tracks, detections, chi2_thr):
    T, D = len(tracks), len(detections)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=bool)
    det_xy = np.zeros((D, 2), dtype=np.float32)
    for j, d in enumerate(detections):
        tlwh = d.tlwh
        det_xy[j, 0] = tlwh[0] + tlwh[2] * 0.5
        det_xy[j, 1] = tlwh[1] + tlwh[3] * 0.5
    gate = np.zeros((T, D), dtype=bool)
    for i, t in enumerate(tracks):
        m = t.mean[:2]
        S = t.covariance[:2, :2]
        try: Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError: Sinv = np.linalg.pinv(S)
        dxy = det_xy - m[None, :]
        maha = np.einsum('nd,dd,nd->n', dxy, Sinv, dxy)
        gate[i, :] = (maha <= chi2_thr)
    return gate

# [REVIVE] IoU cost between tracks (predicted state already in .mean) and detection STracks
def _iou_cost_tracks_vs_dets(tracks, detections):
    T, D = len(tracks), len(detections)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)
    C = np.ones((T, D), dtype=np.float32)
    det_tlbr = np.stack([d.tlbr for d in detections], axis=0) if D else None
    for i, tr in enumerate(tracks):
        tr_tlbr = tr.tlbr
        for j in range(D):
            # cost = 1 - IoU
            C[i, j] = 1.0 - iou_single(tr_tlbr, det_tlbr[j])
    return C


def _oks_cost_tracks_vs_dets(tracks, detections, sigmas=None, area_mode="det", eps=1e-6):
    """
    OKS cost = 1 - OKS, lower is better.
    tracks: list[STrack] (pose in track.last_pose)
    detections: list[STrack] (pose in det.last_pose)
    sigmas: None | float | np.ndarray with shape (K,)
    area_mode: "det" | "track" | "min" | "mean"
    """
    T, D = len(tracks), len(detections)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float32)

    # determine K (allow missing poses)
    K = None
    for t in tracks:
        if getattr(t, "last_pose", None) is not None:
            K = t.last_pose.shape[0]; break
    if K is None:
        for d in detections:
            if getattr(d, "last_pose", None) is not None:
                K = d.last_pose.shape[0]; break
    if K is None:
        return np.ones((T, D), dtype=np.float32)  # no pose anywhere → neutral high cost

    if sigmas is None:
        sigmas = 0.05  # a reasonable default; tune per dataset
    if np.isscalar(sigmas):
        sig = float(sigmas)
        sig_vec = np.full((K,), sig, dtype=np.float32)
    else:
        sig_vec = np.asarray(sigmas, dtype=np.float32).reshape(-1)
        if sig_vec.shape[0] != K:
            raise ValueError(f"sigmas length {sig_vec.shape[0]} != K {K}")
    sig2 = (2.0 * sig_vec)**2  # shape (K,)

    C = np.ones((T, D), dtype=np.float32)
    for i, tr in enumerate(tracks):
        p1 = getattr(tr, "last_pose", None)
        if p1 is None or p1.shape[0] != K:
            continue
        a_tr = max(tr.tlwh[2] * tr.tlwh[3], eps)
        for j, det in enumerate(detections):
            p2 = getattr(det, "last_pose", None)
            if p2 is None or p2.shape[0] != K:
                continue
            a_det = max(det.tlwh[2] * det.tlwh[3], eps)
            if area_mode == "det":
                a = a_det
            elif area_mode == "track":
                a = a_tr
            elif area_mode == "min":
                a = min(a_tr, a_det)
            else:
                a = 0.5 * (a_tr + a_det)

            # per-keypoint squared distance
            dx2 = ((p1 - p2) ** 2).sum(axis=1).astype(np.float32)  # (K,)
            oks_per_k = np.exp(-dx2 / (2.0 * a * sig2 + eps))      # (K,)
            oks = float(oks_per_k.mean())
            C[i, j] = 1.0 - oks
    return C