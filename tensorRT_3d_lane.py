#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt
import time

import matplotlib.pyplot as plt
import math

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

CLASSES = ['Pedestrian', 'Car', 'Cyclist']

class_num = len(CLASSES)
center_input_h = 384
center_input_w = 1280
lane_input_h = 288
lane_input_w = 800

object_thresh = 0.6

lane_output_w = 200

output_h = 96
output_w = 320
downsample_ratio = 4
num_heading_bin = 12

BEV_WIDTH = 700   # 图片像素宽
BEV_HEIGHT = 600 # 图片像素高
X_RANGE = 35      # 左右范围 ±35m
Z_RANGE = 60      # 前方 60m

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class ScoreXY:
    def __init__(self, score, c, h, w):
        self.score = score
        self.c = c
        self.h = h
        self.w = w


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

def center_precess_image(img_src, resize_w, resize_h):
    img_letter, scale, pad_w, pad_h = letterbox(img_src, resize_w, resize_h)
    image = cv2.cvtColor(img_letter, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image /= 255.0
    image -= mean
    image /= std

    image = image.transpose((2, 0, 1))
    image = np.ascontiguousarray(image)
    return image, scale, pad_w, pad_h

def lane_precess_image(img_src):
    img_letter, scale, pad_w, pad_h = letterbox(img_src, 800, 288)

    img = cv2.cvtColor(img_letter, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img, scale, pad_w, pad_h

def nms(heatmap, heatmapmax):
    keep_heatmap = []
    for b in range(1):
        for c in range(class_num):
            for h in range(output_h):
                for w in range(output_w):
                    if heatmapmax[c * output_h * output_w + h * output_w + w] == heatmap[
                        c * output_h * output_w + h * output_w + w] and heatmap[
                        c * output_h * output_w + h * output_w + w] > object_thresh:
                        temp = ScoreXY(heatmap[c * output_h * output_w + h * output_w + w], c, h, w)
                        keep_heatmap.append(temp)
    return keep_heatmap


def sigmoid(x):
    return 1 / (1 + exp(-x))


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


class Calibration(object):
    def __init__(self, calib_file):
        with open(calib_file) as f:
            lines = f.readlines()
        obj = lines[2].strip().split(' ')[1:]
        self.P2 = np.array(obj, dtype=np.float32).reshape(3, 4)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def img_to_rect(self, u, v, depth_rect):
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = [x, y, depth_rect]
        return pts_rect

    def alpha2ry(self, alpha, u):
        ry = alpha + np.arctan2(u - self.cu, self.fu)
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry

def lane_postprocess(output):
    result = np.zeros(shape=(18, 4))
    for i in range(4):
        for j in range(18):
            total = 0
            maxvalue = 0
            maxindex = 0
            for k in range(200):
                if maxvalue < output[k, j, i]:
                    maxvalue = output[k, j, i]
                    maxindex = k
                if k == 199:
                    if maxvalue < output[k + 1, j, i]:
                        maxvalue = output[k + 1, j, i]
                        maxindex = k

                tmp = exp(output[k, j, i])
                total += tmp

            for k in range(200):
                if maxindex < 199:
                    tmp = exp(output[k, j, i]) / total
                    output[k, j, i] = tmp
                    result[17 - j, i] += tmp * (k + 1)

    return result

def center_postprocess(outputs, calibs):
    heatmapmax = outputs[0]
    heatmap = outputs[1]
    offset_2d = outputs[2]
    size_2d = outputs[3]

    depths = outputs[4]
    offset_3d = outputs[5]
    size_3d = outputs[6]
    heading = outputs[7]

    keep_heatmap = nms(heatmap, heatmapmax)
    top_heatmap = sorted(keep_heatmap, key=lambda t: t.score, reverse=True)

    boxes2d = []
    output3d = []

    for i in range(len(top_heatmap)):
        if i > 50:
            break
        classId = top_heatmap[i].c
        score = top_heatmap[i].score
        w = top_heatmap[i].w
        h = top_heatmap[i].h

        # 解码 2d 框
        bx = (w + offset_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        by = (h + offset_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bw = (size_2d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
        bh = (size_2d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio

        xmin = (bx - bw / 2) / center_input_w
        ymin = (by - bh / 2) / center_input_h
        xmax = (bx + bw / 2) / center_input_w
        ymax = (by + bh / 2) / center_input_h

        keep_flag = 0
        for j in range(len(boxes2d)):
            xmin1 = boxes2d[j].xmin
            ymin1 = boxes2d[j].ymin
            xmax1 = boxes2d[j].xmax
            ymax1 = boxes2d[j].ymax
            if IOU(xmin, ymin, xmax, ymax, xmin1, ymin1, xmax1, ymax1) > 0.45:
                keep_flag += 1
                break

        if keep_flag == 0:
            bbox = DetectBox(classId, score, xmin, ymin, xmax, ymax)
            boxes2d.append(bbox)

            # 解码 3DBox
            dimensions = []
            headings = []

            depth = depths[0 * output_h * output_w + h * output_w + w]
            sigma = depths[1 * output_h * output_w + h * output_w + w]

            depth = 1. / (sigmoid(depth) + 1e-6) - 1.
            sigma = np.exp(-sigma)

            x3d = (w + offset_3d[0 * output_h * output_w + h * output_w + w]) * downsample_ratio
            y3d = (h + offset_3d[1 * output_h * output_w + h * output_w + w]) * downsample_ratio

            dimensions.append(size_3d[0 * output_h * output_w + h * output_w + w])
            dimensions.append(size_3d[1 * output_h * output_w + h * output_w + w])
            dimensions.append(size_3d[2 * output_h * output_w + h * output_w + w])

            for k in range(24):
                headings.append(heading[k * output_h * output_w + h * output_w + w])

            locations = calibs.img_to_rect(x3d, y3d, depth)
            locations[1] += dimensions[0] / 2
            alpha = get_heading_angle(headings)
            ry = calibs.alpha2ry(alpha, x3d)

            # 结果输出
            output3d.append(score)
            output3d.append(classId)
            output3d.append(alpha)

            # 理论上这里的 xmin, ymin,xmax, ymax 用 3D 结果计算(这里直接用的 2d 解码的框)
            output3d.append(xmin)
            output3d.append(ymin)
            output3d.append(xmax)
            output3d.append(ymax)

            output3d.append(dimensions[0])
            output3d.append(dimensions[1])
            output3d.append(dimensions[2])

            output3d.append(locations[0])
            output3d.append(locations[1])
            output3d.append(locations[2])

            output3d.append(ry)

    return boxes2d, output3d


def class2angle(cls, residual, to_label_format=False):
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)


class Object3d(object):
    def __init__(self, data):
        # extract label, truncation, occlusion
        self.score = data[0]  # score
        self.type = data[1]  # 'Car', 'Pedestrian', ...
        self.alpha = data[2]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[3]  # left
        self.ymin = data[4]  # top
        self.xmax = data[5]  # right
        self.ymax = data[6]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[7]  # box height
        self.w = data[8]  # box width
        self.l = data[9]  # box length (in meters)
        self.t = (data[10], data[11], data[12])  # location (x,y,z) in camera coord.
        self.ry = data[13]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]


def project_to_image(pts_3d, P):
    '''
    将相机坐标系下的3D边界框的角点, 投影到图像平面上, 得到它们在图像上的2D坐标
    输入: pts_3d是一个nx3的矩阵, 包含了待投影的3D坐标点(每行一个点), P是相机的投影矩阵, 通常是一个3x4的矩阵。
    输出: 返回一个nx2的矩阵, 包含了投影到图像平面上的2D坐标点。
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)  => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)   => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]  # 获取3D点的数量
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))  # 将每个3D点的坐标扩展为齐次坐标形式（4D），通过在每个点的末尾添加1，创建了一个nx4的矩阵。
    pts_2d = np.dot(pts_3d_extend,
                    np.transpose(P))  # 将扩展的3D坐标点矩阵与投影矩阵P相乘，得到一个nx3的矩阵，其中每一行包含了3D点在图像平面上的投影坐标。每个点的坐标表示为[x, y, z]。
    pts_2d[:, 0] /= pts_2d[:, 2]  # 将投影坐标中的x坐标除以z坐标，从而获得2D图像上的x坐标。
    pts_2d[:, 1] /= pts_2d[:, 2]  # 将投影坐标中的y坐标除以z坐标，从而获得2D图像上的y坐标。
    return pts_2d[:, 0:2]  # 返回一个nx2的矩阵,其中包含了每个3D点在2D图像上的坐标。


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d(obj, P):
    '''
    计算对象的3D边界框在图像平面上的投影
    输入: obj代表一个物体标签信息,  P代表相机的投影矩阵-内参。
    输出: 返回两个值, corners_3d表示3D边界框在 相机坐标系 的8个角点的坐标-3D坐标。
                                     corners_2d表示3D边界框在 图像上 的8个角点的坐标-2D坐标。
    '''
    # 计算一个绕Y轴旋转的旋转矩阵R，用于将3D坐标从世界坐标系转换到相机坐标系。obj.ry是对象的偏航角
    R = roty(obj.ry)

    # 物体实际的长、宽、高
    l = obj.l
    w = obj.w
    h = obj.h

    # 存储了3D边界框的8个角点相对于对象中心的坐标。这些坐标定义了3D边界框的形状。
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # 1、将3D边界框的角点坐标从对象坐标系转换到相机坐标系。它使用了旋转矩阵R
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # 3D边界框的坐标进行平移
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

    # 2、检查对象是否在相机前方，因为只有在相机前方的对象才会被绘制。
    # 如果对象的Z坐标（深度）小于0.1，就意味着对象在相机后方，那么corners_2d将被设置为None，函数将返回None。
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # 3、将相机坐标系下的3D边界框的角点，投影到图像平面上，得到它们在图像上的2D坐标。
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)


def reshape_results(results3d_flat):
    OBJ_LEN = 14
    assert len(results3d_flat) % OBJ_LEN == 0, "results3d length mismatch!"

    results = []
    num_obj = len(results3d_flat) // OBJ_LEN
    for i in range(num_obj):
        start = i * OBJ_LEN
        results.append(results3d_flat[start:start + OBJ_LEN])
    return results

def draw_bev_cv(results3d):
    # 创建空白 BEV 图
    bev_img = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

    # 中心点像素坐标
    center_x = BEV_WIDTH // 2
    center_z = BEV_HEIGHT  # 车辆在底部

    # 绘制坐标轴
    cv2.arrowedLine(bev_img, (center_x, center_z), (center_x + 50, center_z), (255,0,0), 2)
    cv2.arrowedLine(bev_img, (center_x, center_z), (center_x, center_z - 50), (0,255,0), 2)

    for obj in results3d:
        score = obj[0]
        cls_id = obj[1]
        h3d, w3d, l3d = obj[7], obj[9], obj[8]
        x3d, y3d, z3d = obj[10], obj[11], obj[12]
        yaw = obj[13]

        # 局部角点
        corners_local = np.array([
            [-w3d/2, -l3d/2],
            [ w3d/2, -l3d/2],
            [ w3d/2,  l3d/2],
            [-w3d/2,  l3d/2]
        ])
        rot_matrix = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]
        ])
        corners_rotated = corners_local.dot(rot_matrix.T)
        corners_global = corners_rotated + np.array([x3d, z3d])

        # 坐标映射到像素
        pts = []
        for px, pz in corners_global:
            u = int(center_x + (px / X_RANGE) * (BEV_WIDTH / 2))
            v = int(center_z - (pz / Z_RANGE) * BEV_HEIGHT)
            pts.append((u, v))
        pts.append(pts[0])  # 闭合

        # 绘制矩形
        for i in range(4):
            cv2.line(bev_img, pts[i], pts[i+1], (0,0,255), 2)
        # 绘制文本
        text_pos = ((pts[0][0] + pts[2][0])//2, (pts[0][1] + pts[2][1])//2)
        cv2.putText(bev_img, f"{cls_id}:{score:.2f}", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

    # 显示 BEV
    cv2.imshow("BEV", bev_img)
    cv2.waitKey(0)


def letterbox(img, new_w=800, new_h=288, color=(114, 114, 114)):
    h, w = img.shape[:2]

    # 等比例缩放
    scale = min(new_w / w, new_h / h)
    resized_w = int(w * scale)
    resized_h = int(h * scale)

    # 缩放
    img_resized = cv2.resize(img, (resized_w, resized_h))

    # 创建新图（目标大小）
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)

    # 剩余填充（居中）
    pad_w = (new_w - resized_w) // 2
    pad_h = (new_h - resized_h) // 2

    # 放到 canvas 中心
    canvas[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = img_resized

    return canvas, scale, pad_w, pad_h


def main():
    centerNet3D_engine_file_path = './weights/Monodle_centerNet3D_fp32.trt'
    FastLane_engine_file_path = './weights/UltraFastLaneDetection_fp32.trt'
    image_path = './images/test6.png'
    cal_path = './test.txt'

    origin_image = cv2.imread(image_path)
    image_h, image_w = origin_image.shape[0:2]
    center_image, center_scale, center_pad_w, center_pad_h = center_precess_image(origin_image, center_input_w, center_input_h)
    lane_image, lane_scale, lane_pad_w, lane_pad_h = lane_precess_image(origin_image)

    # ====== 加载 CenterNet3D 模型 ======
    center_engine = get_engine_from_bin(centerNet3D_engine_file_path)
    center_context = center_engine.create_execution_context()
    center_inputs, center_outputs, center_bindings, center_stream = allocate_buffers(center_engine)

    # ====== 加载车道线模型 ======
    lane_engine = get_engine_from_bin(FastLane_engine_file_path)
    lane_context = lane_engine.create_execution_context()
    lane_inputs, lane_outputs, lane_bindings, lane_stream = allocate_buffers(lane_engine)

    # ==== CenterNet3D  推理 ====
    center_inputs[0].host = center_image
    center_trt_outputs = do_inference(center_context, bindings=center_bindings, inputs=center_inputs, outputs=center_outputs, stream=center_stream,
                                batch_size=1)

    center_outputs = []
    for i in range(len(center_trt_outputs)):
        center_outputs.append(center_trt_outputs[i].reshape(-1))

    calibs = Calibration(cal_path)
    boxes2d, results3d = center_postprocess(center_outputs, calibs)


    # ==== Lane Detection 推理 ====
    np.copyto(lane_inputs[0].host, lane_image.ravel())
    lane_trt_outputs = do_inference(lane_context, bindings=lane_bindings, inputs=lane_inputs, outputs=lane_outputs, stream=lane_stream,
                                batch_size=1)

    lane_trt_out = lane_trt_outputs[0].reshape(1, 201, 18, 4)
    lane_result = lane_postprocess(lane_trt_out[0])


    # 画 3D 框
    for l in range(0, len(results3d), 14):
        obj3d = []
        for q in range(14):
            obj3d.append(results3d[l + q])
        obj3ds = Object3d(obj3d)
        box3d, box3d_pts_3d = compute_box_3d(obj3ds, calibs.P2)

        for k in range(0, 4):
            # 上底面
            i, j = k, (k + 1) % 4
            x1, y1 = int((box3d[i, 0] - center_pad_w) / center_scale), int((box3d[i, 1] - center_pad_h) / center_scale)
            x2, y2 = int((box3d[j, 0] - center_pad_w) / center_scale), int((box3d[j, 1] - center_pad_h) / center_scale)
            cv2.line(origin_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 下底面
            i, j = k + 4, (k + 1) % 4 + 4
            x1, y1 = int((box3d[i, 0] - center_pad_w) / center_scale), int((box3d[i, 1] - center_pad_h) / center_scale)
            x2, y2 = int((box3d[j, 0] - center_pad_w) / center_scale), int((box3d[j, 1] - center_pad_h) / center_scale)
            cv2.line(origin_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 竖边
            i, j = k, k + 4
            x1, y1 = int((box3d[i, 0] - center_pad_w) / center_scale), int((box3d[i, 1] - center_pad_h) / center_scale)
            x2, y2 = int((box3d[j, 0] - center_pad_w) / center_scale), int((box3d[j, 1] - center_pad_h) / center_scale)
            cv2.line(origin_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for i in range(len(boxes2d)):
        classid = boxes2d[i].classId
        score = boxes2d[i].score
        xmin_model = boxes2d[i].xmin * center_input_w
        ymin_model = boxes2d[i].ymin * center_input_h
        xmax_model = boxes2d[i].xmax * center_input_w
        ymax_model = boxes2d[i].ymax * center_input_h
        xmin = int((xmin_model - center_pad_w) / center_scale)
        ymin = int((ymin_model - center_pad_h) / center_scale)
        xmax = int((xmax_model - center_pad_w) / center_scale)
        ymax = int((ymax_model - center_pad_h) / center_scale)


        cv2.rectangle(origin_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = '%s:%.2f' % (CLASSES[classid], score)
        cv2.putText(origin_image, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    for lane_idx in range(lane_result.shape[1]):
        for row in range(lane_result.shape[0]):
            if lane_result[row, lane_idx] > 0:

                x_model = (lane_result[row, lane_idx] - 1) * (lane_input_w - 1) / (lane_output_w - 1)

                y_model = 288 - 1 - row * ((170 - 1) / (18 - 1))

                # 映射到原图
                x_img = int((x_model - lane_pad_w) / lane_scale)
                y_img = int((y_model - lane_pad_h) / lane_scale)

                if x_img < 0 or x_img >= image_w:
                    continue
                if y_img < 0 or y_img >= image_h:
                    continue

                color = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)][lane_idx]
                cv2.circle(origin_image, (x_img, y_img), 5, color, -1)

    cv2.imshow("3D", origin_image)
    cv2.waitKey(0)

    K = calibs.P2[:3,:3]
    results = reshape_results(results3d)
    draw_bev_cv(results)

        


if __name__ == '__main__':
    print('This is main ...')
    main()
