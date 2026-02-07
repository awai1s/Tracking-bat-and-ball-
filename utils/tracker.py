# tracker.py
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

class ObjectTracker:
    """
    Tracks objects (cricket ball) using IoU-first matching, distance fallback,
    and Lucas-Kanade optical flow as a short-term predictor when detections miss.
    Call update(detections, frame_gray) to enable optical-flow fallback.
    """

    def __init__(
        self,
        max_lost: int = 8,
        distance_threshold: float = 120.0,
        iou_threshold: float = 0.3,
        frame_size: Optional[Tuple[int, int]] = None
    ):
        self.next_object_id = 0
        self.objects: Dict[int, Dict] = {}   # object_id -> {'bbox','centroid'}
        self.lost: Dict[int, int] = {}
        self.max_lost = int(max_lost)
        self.distance_threshold = float(distance_threshold)
        self.iou_threshold = float(iou_threshold)
        self.frame_size = frame_size

        # Optical flow state
        self.prev_gray = None
        self.prev_pts: Dict[int, np.ndarray] = {}  # object_id -> point array shape (1,1,2)

        # Track how many frames an id has existed (stability)
        self.age: Dict[int, int] = {}

        # LK params
        self.lk_params = dict(winSize=(15,15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    @staticmethod
    def _centroid_from_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        return (cx, cy)

    @staticmethod
    def _iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
        boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
        union = boxAArea + boxBArea - interArea
        if union <= 0:
            return 0.0
        return interArea / union

    def _clamp_bbox(self, bbox: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
        if self.frame_size is None:
            return bbox
        w, h = self.frame_size
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        return (int(x1), int(y1), int(x2), int(y2))

    def register(self, bbox: Tuple[int,int,int,int]) -> None:
        bbox = self._clamp_bbox(bbox)
        centroid = self._centroid_from_bbox(bbox)
        self.objects[self.next_object_id] = {'bbox': bbox, 'centroid': centroid}
        self.lost[self.next_object_id] = 0
        self.age[self.next_object_id] = 0
        # init flow point
        self.prev_pts[self.next_object_id] = np.array([[[centroid[0], centroid[1]]]], dtype=np.float32)
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.lost:
            del self.lost[object_id]
        if object_id in self.prev_pts:
            del self.prev_pts[object_id]
        if object_id in self.age:
            del self.age[object_id]

    def _optical_flow_predict(self, frame_gray: np.ndarray):
        """
        Move each stored prev_pts to a new location using LK flow,
        update centroid if flow succeeded.
        """
        if self.prev_gray is None or frame_gray is None:
            return

        if len(self.prev_pts) == 0:
            return

        # concat points
        pts_ids = list(self.prev_pts.keys())
        pts = np.vstack([self.prev_pts[oid] for oid in pts_ids])  # shape (N,1,2)

        nextPts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, pts, None, **self.lk_params)
        # iterate back
        idx = 0
        for i, oid in enumerate(pts_ids):
            st = status[i][0]
            if st == 1:
                nx, ny = nextPts[i][0]
                self.prev_pts[oid] = np.array([[[nx, ny]]], dtype=np.float32)
                # update centroid and bbox by shifting bbox maintaining size
                if oid in self.objects:
                    old_bbox = self.objects[oid]['bbox']
                    cx_old, cy_old = self.objects[oid]['centroid']
                    w = old_bbox[2] - old_bbox[0]
                    h = old_bbox[3] - old_bbox[1]
                    nx_i, ny_i = int(round(nx)), int(round(ny))
                    new_bbox = (nx_i - w//2, ny_i - h//2, nx_i + (w - w//2), ny_i + (h - h//2))
                    new_bbox = self._clamp_bbox(new_bbox)
                    self.objects[oid]['bbox'] = new_bbox
                    self.objects[oid]['centroid'] = (nx_i, ny_i)
            else:
                # flow failed -> don't change
                pass

    def update(self, detections: List[Tuple[int,int,int,int]], frame_gray: Optional[np.ndarray] = None) -> Dict[int, Dict]:
        """
        detections: list of (x1,y1,x2,y2)
        frame_gray: current frame as grayscale np.ndarray (optional) - used for LK fallback
        Returns mapping object_id -> {'bbox', 'centroid'}
        """
        det_bboxes = [self._clamp_bbox(b) for b in detections]
        det_centroids = [self._centroid_from_bbox(b) for b in det_bboxes]

        # If we have a prev frame, attempt to predict object movement via optical flow
        if frame_gray is not None and self.prev_gray is not None:
            self._optical_flow_predict(frame_gray)

        # If no detections: increment lost and possibly deregister
        if len(det_centroids) == 0:
            to_deregister = []
            for obj_id in list(self.lost.keys()):
                self.lost[obj_id] += 1
                if self.lost[obj_id] > self.max_lost:
                    to_deregister.append(obj_id)
            for oid in to_deregister:
                self.deregister(oid)
            # update prev_gray for next iteration
            self.prev_gray = frame_gray if frame_gray is not None else self.prev_gray
            return self.objects

        # If no existing objects: register all detections
        if len(self.objects) == 0:
            for bbox in det_bboxes:
                self.register(bbox)
            self.prev_gray = frame_gray if frame_gray is not None else self.prev_gray
            return self.objects

        # Build arrays for matching
        object_ids = list(self.objects.keys())
        obj_bboxes = np.array([self.objects[oid]['bbox'] for oid in object_ids], dtype=np.int32)

        # IoU matrix
        iou_matrix = np.zeros((len(object_ids), len(det_bboxes)), dtype=np.float32)
        for i, ob in enumerate(obj_bboxes):
            for j, db in enumerate(det_bboxes):
                iou_matrix[i, j] = self._iou(tuple(ob), tuple(db))

        matched_rows = set()
        matched_cols = set()

        # Primary: IoU greedy
        pairs = []
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for iou_val, row_idx, col_idx in pairs:
            if iou_val < self.iou_threshold:
                continue
            if row_idx in matched_rows or col_idx in matched_cols:
                continue
            obj_id = object_ids[row_idx]
            new_bbox = det_bboxes[col_idx]
            new_centroid = det_centroids[col_idx]
            self.objects[obj_id]['bbox'] = new_bbox
            self.objects[obj_id]['centroid'] = new_centroid
            self.prev_pts[obj_id] = np.array([[[new_centroid[0], new_centroid[1]]]], dtype=np.float32)
            self.lost[obj_id] = 0
            matched_rows.add(row_idx)
            matched_cols.add(col_idx)

        # Fallback: distance-based matching for remaining
        remaining_rows = [r for r in range(len(object_ids)) if r not in matched_rows]
        remaining_cols = [c for c in range(len(det_bboxes)) if c not in matched_cols]

        if remaining_rows and remaining_cols:
            obj_centroids = np.array([self.objects[object_ids[r]]['centroid'] for r in remaining_rows])
            det_centroids_np = np.array([det_centroids[c] for c in remaining_cols])
            D = np.linalg.norm(obj_centroids[:, None, :] - det_centroids_np[None, :, :], axis=2)

            rows_order = D.min(axis=1).argsort()
            cols_assign = D.argmin(axis=1)[rows_order]

            for k, row_local in enumerate(rows_order):
                row_idx = remaining_rows[row_local]
                col_local = cols_assign[k]
                col_idx = remaining_cols[col_local]
                if row_idx in matched_rows or col_idx in matched_cols:
                    continue
                if D[row_local, col_local] > self.distance_threshold:
                    continue
                obj_id = object_ids[row_idx]
                new_bbox = det_bboxes[col_idx]
                new_centroid = det_centroids[col_idx]
                self.objects[obj_id]['bbox'] = new_bbox
                self.objects[obj_id]['centroid'] = new_centroid
                self.prev_pts[obj_id] = np.array([[[new_centroid[0], new_centroid[1]]]], dtype=np.float32)
                self.lost[obj_id] = 0
                matched_rows.add(row_idx)
                matched_cols.add(col_idx)

        # Register unmatched detections
        for det_idx, bbox in enumerate(det_bboxes):
            if det_idx not in matched_cols:
                self.register(bbox)

        # Increment lost for unmatched objects
        for row_idx, obj_id in enumerate(object_ids):
            if row_idx not in matched_rows:
                self.lost[obj_id] += 1
                if self.lost[obj_id] > self.max_lost:
                    self.deregister(obj_id)
            else:
                # reset lost already done above
                pass

        # Increase ages
        for oid in list(self.age.keys()):
            self.age[oid] += 1

        # Store prev_gray and prev points
        self.prev_gray = frame_gray if frame_gray is not None else self.prev_gray

        return self.objects
