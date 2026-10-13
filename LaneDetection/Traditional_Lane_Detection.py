"""传统车道检测：去噪 → Canny → 颜色/ROI 筛选 → Hough → 左右直线拟合。

python Traditional_Lane_Detection.py --input img.jpg --debug-dir outputs/steps
python Traditional_Lane_Detection.py --input video.mp4 --output outputs/traditional.mp4
追加 --show 弹出实时预览窗口；视频按 Q/Esc 或关闭窗口可停止，保存已处理部分。
只估计当前车道的两侧直线；弯道、遮挡、逆光是手工规则的局限。
"""
import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def roi_mask(shape, top=.60):
    """归一化梯形 ROI，摄像机位置改变时需要调整比例。"""
    h, w = shape[:2]
    points = np.array([[(int(.08*w), h-1), (int(.44*w), int(top*h)),
                        (int(.56*w), int(top*h)), (int(.95*w), h-1)]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, points, 255)
    return mask


def fit_lane(segments, h, w, top):
    """按线段长度加权拟合 x=a*y+b，避免竖直线的 dy/dx 不稳定。"""
    if not segments:
        return None
    points, weights = [], []
    for x1, y1, x2, y2 in segments:
        length = np.hypot(x2-x1, y2-y1)
        points.extend([(x1, y1), (x2, y2)])
        weights.extend([length, length])
    points = np.asarray(points, dtype=np.float64)
    design = np.column_stack([points[:, 1], np.ones(len(points))])
    root_weight = np.sqrt(weights)
    a, b = np.linalg.lstsq(design * root_weight[:, None],
                          points[:, 0] * root_weight, rcond=None)[0]
    y_bottom, y_top = h-1, int(top*h)
    xs = [a*y_bottom+b, a*y_top+b]
    if not all(-.1*w <= x <= 1.1*w for x in xs):
        return None
    return (int(np.clip(xs[0], 0, w-1)), y_bottom,
            int(np.clip(xs[1], 0, w-1)), y_top)


def detect_lanes(frame, canny_low=50, canny_high=150, roi_top=.60,
                 use_color=True, close_gaps=False):
    """输入 BGR，返回叠加图、中间结果；无车道时保留原图。"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度图
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 去噪
    edges = cv2.Canny(blurred, canny_low, canny_high) # canny识别边缘
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS) # HLS颜色空间，H:色调，L:亮度，S:饱和度
    white = cv2.inRange(hls, np.array([0, 180, 0]), np.array([180, 255, 255])) # 白色车道线
    yellow = cv2.inRange(hls, np.array([15, 30, 80]), np.array([40, 255, 255])) # 黄色车道线
    colors = cv2.bitwise_or(white, yellow) # 颜色筛选
    selected = cv2.bitwise_and(edges, colors) if use_color else edges.copy() # 颜色+边缘筛选
    # 闭运算可能连接虚线，也可能连接干扰物；默认关闭，供对比实验。
    if close_gaps:
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) # 形态学 闭运算 用途是连接边缘间的间隙
    roi = roi_mask(frame.shape, roi_top) # ROI掩码
    selected = cv2.bitwise_and(selected, roi) # 只保留 ROI 区域
    lines = cv2.HoughLinesP(selected, 1, np.pi/180, threshold=30,
                            minLineLength=max(15, int(.04*h)), maxLineGap=max(1, int(.05*h))) # 霍夫变换检测直线
    groups = {"left": [], "right": []} # 分组：左车道线、右车道线
    raw = frame.copy() 
    if lines is not None:
        # OpenCV 不同版本可能返回 [N,1,4] 或 [N,4]。
        for x1, y1, x2, y2 in lines.reshape(-1, 4): # 遍历检测到的直线
            cv2.line(raw, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if y1 == y2:
                continue
            dx_dy = (x2-x1)/(y2-y1) # 计算斜率
            midpoint = (x1+x2)/2
            if .15 < abs(dx_dy) < 1.5: # 斜率范围，排除竖直线和水平线，避免拟合不稳定。斜率过大或过小的线段可能不是车道线。
                if dx_dy < 0 and midpoint < .55*w:
                    groups["left"].append((x1, y1, x2, y2))
                elif dx_dy > 0 and midpoint > .45*w: # 斜率为正且中点在图像右半部分，认为是右车道线
                    groups["right"].append((x1, y1, x2, y2))
    lane_layer = np.zeros_like(frame) 
    fitted = {}
    for side, segments in groups.items(): # 遍历左右车道线分组
        lane = fit_lane(segments, h, w, roi_top)
        fitted[side] = lane
        if lane is not None:
            x1, y1, x2, y2 = lane
            cv2.line(lane_layer, (x1, y1), (x2, y2), (0, 0, 255), max(2, w//160)) # 绘制拟合的车道线，颜色为红色，线宽根据图像宽度自适应
    overlay = cv2.addWeighted(frame, 1, lane_layer, .85, 0) # 将原图和车道线叠加，alpha=0.85，beta=1，gamma=0
    return overlay, {"gray": gray, "blurred": blurred, "colors": colors,
                     "edges": edges, "roi": roi, "selected_edges": selected,
                     "hough_segments": raw, "lanes": fitted}


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"无法保存图片：{path}")


def process_video(source, output, show=False, **kwargs):
    '''
    处理视频,需要一帧一帧处理，然后保存输出视频；
    show=True 弹出预览窗口，按 Q/Esc 停止。
    '''
    output = Path(output)
    if Path(source).resolve() == output.resolve():
        raise ValueError("输出不能覆盖输入视频")
    cap, writer, count = cv2.VideoCapture(str(source)), None, 0 # VideoCapture 用于读取视频帧，writer 用于写入输出视频，count 记录处理的帧数
    window_name = "Lane Detection - Q/Esc to stop"
    window_created = False
    try:
        if not cap.isOpened(): # 检查视频是否成功打开
            raise OSError(f"无法打开视频：{source}")
        fps = cap.get(cv2.CAP_PROP_FPS) # 获取视频帧率
        fps = fps if np.isfinite(fps) and fps > 0 else 25.
        output.parent.mkdir(parents=True, exist_ok=True)
        while True:
            frame_started = perf_counter() # 记录当前帧开始处理的时间，用于计算处理时间和控制帧率
            ok, frame = cap.read() # 读取视频帧，ok 表示是否成功读取，frame 是当前帧图像
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) # 创建视频写入对象，指定输出路径、编码格式、帧率和帧大小
                if not writer.isOpened():
                    raise OSError(f"无法创建视频：{output}")
            overlay, _ = detect_lanes(frame, **kwargs) # 调用 detect_lanes 函数进行车道检测，返回叠加图 overlay 和中间结果（未使用）
            writer.write(overlay) # 将叠加图写入输出视频
            count += 1 # 处理帧数加一
            if show:
                if not window_created: 
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 创建可调整大小的窗口
                    window_created = True
                cv2.imshow(window_name, overlay) # 实时显示叠加图
                # waitKey 处理窗口事件；减去计算时间，尽量按原视频帧率播放。
                delay = max(1, round(1000/fps - 1000*(perf_counter()-frame_started)))
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        if count == 0:
            raise OSError("视频没有可读取的帧")
    finally:
        cap.release() # 释放视频读取对象
        if writer is not None:
            writer.release()
        if window_created:
            # 用户可能已经手动关闭窗口。
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
    print(f"已处理 {count} 帧，保存到 {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="img.jpg")
    parser.add_argument("--output")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--roi-top", type=float, default=.60)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--close-gaps", action="store_true")
    parser.add_argument("--debug-dir", help="保存图片的中间步骤")
    parser.add_argument("--show", action="store_true", help="弹出预览窗口；视频按 Q/Esc 停止")
    args = parser.parse_args()
    if not 0 < args.roi_top < 1 or not 0 <= args.canny_low < args.canny_high <= 255:
        parser.error("需要 0 < roi-top < 1 和 0 <= canny-low < canny-high <= 255")
    kwargs = dict(canny_low=args.canny_low, canny_high=args.canny_high,
                  roi_top=args.roi_top, use_color=not args.no_color, close_gaps=args.close_gaps)
    is_image = Path(args.input).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    output = args.output or ("outputs/traditional.jpg" if is_image else "outputs/traditional.mp4")
    if Path(args.input).resolve() == Path(output).resolve():
        parser.error("输出不能覆盖输入文件")
    if is_image:
        frame = cv2.imread(args.input)
        if frame is None:
            raise FileNotFoundError(f"无法读取图片：{args.input}")
        overlay, steps = detect_lanes(frame, **kwargs)
        save_image(output, overlay)
        if args.debug_dir:
            for name, value in steps.items():
                if isinstance(value, np.ndarray):
                    save_image(Path(args.debug_dir)/f"{name}.png", value)
        print(f"车道拟合：{steps['lanes']}，结果：{output}")
        if args.show:
            window_name = "Lane Detection - press any key to close"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            try:
                cv2.imshow(window_name, overlay)
                while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                    if cv2.waitKey(50) >= 0:
                        break
            finally:
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass
    else:
        process_video(args.input, output, show=args.show, **kwargs)


if __name__ == "__main__":
    main()


