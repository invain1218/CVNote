"""
Kalman Filter Implementation

Five Equations of the Kalman Filter:

State Prediction:
x̂ₖ|ₖ₋₁ = A * x̂ₖ₋₁|ₖ₋₁ 

Covariance Prediction:
Pₖ|ₖ₋₁ = A * Pₖ₋₁|ₖ₋₁ * Aᵀ + Q

(状态经过线性变换A，协方差作为状态的二阶统计量，必须跟着变)
(Var(x̂ₖ|ₖ₋₁) = A * Var(x̂ₖ₋₁|ₖ₋₁) * Aᵀ + Q)
(Pₖ|ₖ₋₁ = A * Pₖ₋₁|ₖ₋₁ * Aᵀ + Q)
(预测协方差 = 旧协方差经过运动模型传播 + 模型本身的不确定度)


Kalman Gain Calculation:
Kₖ = Pₖ|ₖ₋₁ * Hᵀ * (H * Pₖ|ₖ₋₁ * Hᵀ + R)⁻¹
(Kₖ = Pₖ|ₖ₋₁ *  (H * Pₖ|ₖ₋₁ + R)⁻¹)
(卡尔曼增益 = 预测协方差 * 观测矩阵转置 * (观测矩阵 * 预测协方差 * 观测矩阵转置 + 测量噪声协方差)⁻¹)

State Update:
x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ * (zₖ - H * x̂ₖ|ₖ₋₁)
(后验 = 先验 + 增益 × 新息)

Covariance Update:
Pₖ|ₖ = (I - Kₖ * H) * Pₖ|ₖ₋₁


In this implementation, we will focus on a simple 1D case where we are trying to estimate the position of an object
based on noisy measurements. 
The state vector will consist of the position and velocity of the object.
The Kalman Filter will be implemented in Python using NumPy for matrix operations. 
We will simulate some noisy measurements and apply the Kalman Filter to estimate the true position of the object over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def simulate_data(num_points=100, noise_std=1.0):
    """
    Simulate a simple linear motion with Gaussian noise.
    """
    t = np.linspace(0, 10, num_points)
    true_weight = 0.5 * t**2  # True weight follows a quadratic function
    noise = np.random.normal(0, noise_std, size=num_points)
    measured_weight = true_weight + noise
    return t, true_weight, measured_weight

sim = simulate_data(num_points=100, noise_std=5.0)
plt.figure(figsize=(10, 5))
plt.plot(sim[0], sim[1], label='True Weight', color='g')
plt.scatter(sim[0], sim[2], label='Measured Weight', color='r', s=10)
plt.title('Simulated Data with Noise')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.legend()
plt.show()

def kalman_filter(data, x0, P0, A, H, Q, R):
    """
    Performs Kalman filtering on 1 state variable.
    'data' contains the data to be filtered.
    'x0' is the initial state estimate
    'P0' is the initial covariance estimate
    'A' is the state transition matrix
    'H' is the observation matrix
    'Q' is the process noise covariance
    'R' is the measurement noise covariance
    
    注意，Q是过程噪声协方差，R是测量噪声协方差，这两个值的大小会影响滤波器的性能。Q越大，滤波器对模型的不确定性越敏感；R越大，滤波器对观测噪声越敏感。
    在 SORT 系列中，Q 和 R 是需要手动调整的超参数，并用验证集上的跟踪指标来评估调整效果。
    """
    x_est = x0
    P_est = P0
    results = []
    
    for z in data:
        # Prediction step
        x_pred = A @ x_est  # Predict the next state
        P_pred = A @ P_est @ A.T + Q  # Predict the next covariance

        # Update step
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)  # Kalman Gain
        
        x_est = x_pred + K @ (z - H @ x_pred)  # Update estimate with measurement
        
        P_est = (np.eye(len(K)) - K @ H) @ P_pred  # Update covariance

        results.append(x_est[0])  # Store only the position estimate
    
    return np.array(results)

kalman_results = kalman_filter(sim[2], x0=np.array([[0], [0]]), P0=np.eye(2), 
                               A=np.array([[1, 1], [0, 1]]), H=np.array([[1, 0]]), 
                               Q=np.array([[1, 0], [0, 1]]), R=np.array([[25]]))
plt.figure(figsize=(10, 5))
plt.plot(sim[0], sim[1], label='True Weight', color='g')
plt.scatter(sim[0], sim[2], label='Measured Weight', color='r', s=10)
plt.plot(sim[0], kalman_results, label='Kalman Filtered Weight', color='b')
plt.title('Kalman Filter Results')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.legend()
plt.show()  

"""

"""

# 数据结构：bbox = [x1, y1, x2, y2]  # 左上角和右下角坐标，30fps视频，1秒输出 (30,4) 的检测框数据
# 数据结构需要变换为 \mathbf{x} = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^\top，因此，状态数据（30，7）
# 中心位置 + 尺寸 + 速度

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


init_bbox = np.array([10, 20, 110, 220], dtype=float)
tracker = KalmanBoxTracker(init_bbox, dt=1.0/30.0)
# 模拟 N 帧：目标以 (5+x, 3+y) 速度移动（不存在变形）

N = 3000
H, W = 1080, 1920
vx = 5
vy = 3
trajectory = []
for t in range(N):
    x_move = np.random.normal(1,1) * 0.5
    y_move = np.random.normal(1,1) * 0.5
    det = init_bbox + np.array([
        vx * t + x_move,
        vy * t + y_move,
        vx * t + x_move,
        vy * t + y_move
    ])

    # 加观测噪声
    det = det + np.random.randn(4) * 2

    # kalman 预测
    pred_bbox = tracker.predict()
    # kalman更新
    tracker.update(det)

    # iou score
    iou_score = iou(pred_bbox, det)

    print(
        f"t={t:2d}  "
        f"pred={pred_bbox.round(1)}  "
        f"det={det.round(1)}  "
        f"est={tracker.bbox.round(1)}  "
        f"iou={iou_score:.2f}"
    )

    # 创建画布
    canvas = np.ones((H, W, 3),dtype=np.uint8) * 255

    # 画 bbox
    # detection：蓝色
    x1, y1, x2, y2 = det.astype(int)
    cv2.rectangle(canvas,(x1, y1),(x2, y2),(255, 0, 0),2) # color: blue, thickness: 2
    cv2.putText(canvas,"Detection",(x1, max(y1 - 8, 15)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0), 1) # FONT_HERSHEY_SIMPLEX（简体中文（大陆））

    # prediction：绿色
    x1, y1, x2, y2 = pred_bbox.astype(int)
    cv2.rectangle( canvas,(x1, y1),(x2, y2),(0, 255, 0),2)
    cv2.putText(canvas,"Prediction",(x1, max(y1 - 8, 15)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 150, 0),1)

    # estimated：红色
    est_bbox = tracker.bbox
    x1, y1, x2, y2 = est_bbox.astype(int) # prediction后根据observation校正后的est
    cv2.rectangle(canvas,(x1, y1), (x2, y2),(0, 0, 255),2)
    cv2.putText(canvas,"Kalman Estimate",(x1, min(y2 + 20, H - 5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)

    # 记录轨迹
    x1, y1, x2, y2 = est_bbox.astype(int)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    trajectory.append((cx, cy))
    if t > 30:
        cv2.line(canvas, trajectory[t - 30],trajectory[t],(0, 0, 255), 2)

    # 显示帧信息
    cv2.putText(canvas,f"Frame: {t}",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 0),2)
    cv2.putText(canvas,f"IoU: {iou_score:.2f}", (10, 50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 0),2)

    # 弹出窗口
    cv2.imshow("Kalman Filter Tracking", canvas)
    # 30 FPS，大约33ms一帧
    key = cv2.waitKey(33) & 0xFF
    # 按 q 退出
    if key == ord("q"):
        break

cv2.destroyAllWindows()


