
# 模拟多目标追踪
# 匈牙利算法来解决数据关联问题
#              Track1 Track2 Track3

# Detection1    0.82   0.20   0.05
# Detection2    0.10   0.76   0.20
# Detection3    0.05   0.15   0.91

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def state_to_bbox(state):
    """(u, v, s, r) -> (x1, y1, x2, y2)"""
    u, v, s, r = state[:4]
    w = np.sqrt(s * r)
    h = np.sqrt(s / r)
    return np.array([u - w/2, v - h/2, u + w/2, v + h/2])

def bbox_to_state(bbox, du=0.0, dv=0.0, ds=0.0):
    """(x1, y1, x2, y2) -> [u, v, s, r, du, dv, ds]"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2
    s = w * h
    r = w / h
    return np.array([u, v, s, r, du, dv, ds], dtype=float)

def iou(bbox_a, bbox_b):
    """
    计算两个 bbox 的 IoU
    IoU(A,B)=Area(A∩B)/Area(A∪B)
    交集面积除以并集面积
    这个就造成re-ID的问题，因为是面积，bbox重叠但是可能不是一个object
    """
    xa1, ya1, xa2, ya2 = bbox_a
    xb1, yb1, xb2, yb2 = bbox_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)

class KalmanBoxTracker:
    def __init__(self, bbox, dt=1.0/30.0):
        self.dt = dt # 时间间隔 1/fps
        self.x = bbox_to_state(bbox) # 当前状态向量 [u, v, s, r, du, dv, ds]，7×1 列向量
        self.P = np.eye(7) * 10.0  # 初始协方差 7×7，每个维度的不确定性大概是10，初始化参数 后面会不断更新
        self.A = self._build_A(dt) # 状态转移矩阵（motion）
        self.H = np.eye(4, 7) # 行数 = 观测维度，列数 = 状态维度 纯维度匹配，因为需要把状态向量映射到观测空间，观测空间只有前4个维度（u, v, s, r），所以H是4x7矩阵
        self.Q = self._build_Q() # 过程噪声
        self.R = np.eye(4) * 1.0 # 观测噪声 协方差 噪音正态分布N~(0,1), 默认不确定性为1.0
        
        # MOT        
        self.id = 0                     # 轨迹 ID
        self.hits = 1                   # 累计匹配次数
        self.time_since_update = 0      # 距上次匹配的帧数

    def _build_A(self, dt):
        A = np.eye(7)
        A[0, 4] = dt   # u  <- du
        A[1, 5] = dt   # v  <- dv
        A[2, 6] = dt   # s  <- ds
        return A

    def _build_Q(self):
        # SORT 原始实现的设置
        Q = np.eye(7)
        Q[4:, 4:] *= 0.01   # 速度项噪声更大
        Q[:4, :4] *= 0.0001 # 位置项噪声更小
        return Q

    def predict(self):
        """① 状态预测 ② 协方差预测"""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return state_to_bbox(self.x)

    def update(self, bbox):
        """③ 增益 ④ 状态更新 ⑤ 协方差更新"""
        z = bbox_to_state(bbox)[:4]           # 4x1 观测

        # ③ Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # ④ 状态更新
        y = z - self.H @ self.x  # residual
        self.x = self.x + K @ y 

        # ⑤ 协方差更新
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P

    @property
    def bbox(self):
        return state_to_bbox(self.x) 

class MultiObjectTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3, dt=1.0):
        """
        iou_threshold : 匹配的最低 IoU，低于此值视为未匹配
        max_age       : 轨迹连续多少帧未匹配后删除
        min_hits      : 新轨迹连续多少帧匹配后才确认
        dt            : 帧间隔
        """
        self.trackers = []          # 所有轨迹
        self.next_id = 1            # 轨迹 ID 分配
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.dt = dt
        self.frame_count = 0


    def update(self, detections):
        """
        detections: (N, 4) array of [x1, y1, x2, y2]
        returns   : list of (track_id, bbox)
        """
        self.frame_count += 1

        pred_bboxes = []
        for trk in self.trackers:
            pred_bboxes.append(trk.predict())
            
        # iou矩阵
        n_trk = len(self.trackers)
        n_det = len(detections)
        iou_matrix = np.zeros((n_det, n_trk))

        for d, det in enumerate(detections):
            for t, pred in enumerate(pred_bboxes):
                iou_matrix[d, t] = iou(det, pred)

        # 匈牙利匹配 
        # 代价矩阵 = 1 - IoU，匈牙利算法最小化代价
        cost_matrix = 1 - iou_matrix
        det_idx, trk_idx = linear_sum_assignment(cost_matrix)

        # 过滤低 IoU 的匹配
        matched_det = set()
        matched_trk = set()
        for d, t in zip(det_idx, trk_idx):
            if iou_matrix[d, t] >= self.iou_threshold:
                self.trackers[t].update(detections[d])
                self.trackers[t].hits += 1
                self.trackers[t].time_since_update = 0
                matched_det.add(d)
                matched_trk.add(t)

        # 未匹配的检测框 → 新建轨迹 
        for d in range(n_det):
            if d not in matched_det:
                trk = KalmanBoxTracker(detections[d], dt=self.dt)
                trk.id = self.next_id
                self.next_id += 1
                self.trackers.append(trk)

        # 未匹配的轨迹 → 标记丢失
        for t, trk in enumerate(self.trackers):
            if t not in matched_trk:
                trk.time_since_update += 1

        # 删除超过 max_age 的轨迹
        self.trackers = [
            trk for trk in self.trackers
            if trk.time_since_update <= self.max_age
        ]

        # 输出确认的轨迹
        results = []
        for trk in self.trackers:
            # 连续 min_hits 帧匹配上，或者已经确认过的轨迹
            if trk.hits >= self.min_hits or trk.time_since_update == 0:
                results.append((trk.id, trk.bbox))
        return results





N = 3000
H, W = 1080, 1920
mot = MultiObjectTracker(iou_threshold=0.3, max_age=5, min_hits=1, dt=1.0)

# 每个 track_id 维护一条轨迹历史
trajectories = {}   # {track_id: [(cx, cy), ...]}

# 颜色表：按 ID 分配固定颜色，方便区分不同目标
COLORS = [
    (255, 0, 0),    # 蓝
    (0, 200, 0),    # 绿
    (0, 0, 255),    # 红
    (255, 0, 255),  # 紫
    (0, 165, 255),  # 橙
    (255, 255, 0),  # 青
]

for t in range(N):
    # 模拟检测框
    dets = np.array([
        [10 + 5*t, 20 + 3*t, 110 + 5*t, 220 + 3*t],   # 目标1
        [200 + 2*t, 50 + 4*t, 300 + 2*t, 150 + 4*t],  # 目标2
        [400 + 3*t, 80 + 2*t, 500 + 3*t, 180 + 2*t],  # 目标3
    ], dtype=float)
    dets += np.random.randn(*dets.shape) * 2

    # MOT 更新 
    results = mot.update(dets)   # list of (track_id, est_bbox)

    # 创建画布 
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # 画检测框（蓝色）
    for det in dets:
        x1, y1, x2, y2 = det.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(canvas, "Det", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 画每条轨迹（按 ID 上色）
    for track_id, est_bbox in results:
        color = COLORS[track_id % len(COLORS)]

        # 估计框
        x1, y1, x2, y2 = est_bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"ID{track_id}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 中心点
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # 记录轨迹
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))

        # 画轨迹尾巴
        pts = trajectories[track_id]
        for i in range(1, min(len(pts), 30)):
            cv2.line(canvas, pts[-i], pts[-i-1], color, 2)

    # 帧信息
    cv2.putText(canvas, f"Frame: {t}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, f"Tracks: {len(results)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("MOT Tracking", canvas)
    key = cv2.waitKey(33) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()