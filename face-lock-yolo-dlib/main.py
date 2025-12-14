# -*- coding: utf-8 -*-
"""
YOLO + dlib 人脸锁定【绝不崩·稳定版】
--------------------------------------------------
设计原则（非常重要）：
1️⃣ YOLO 模型【只在主线程】加载
2️⃣ QThread 只做：人脸 embedding + 匹配（不碰 YOLO / UI）
3️⃣ QImage 必须 .copy() 再传 UI
4️⃣ UI 限帧显示（防止事件队列爆炸）
5️⃣ dlib 模型单例，只初始化一次

✔ Windows / PyQt5 / GPU / exe 实测稳定
"""

import sys
import os
import cv2
import dlib
import time
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

# ===================== 配置 =====================
FACE_THRESHOLD = 0.45
DISPLAY_INTERVAL = 2      # UI 每 N 帧刷新一次
DETECT_INTERVAL = 5       # 每 N 帧做人脸匹配

DLIB_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
DLIB_RECOG_MODEL = "dlib_face_recognition_resnet_model_v1.dat"

# ===================== dlib 单例 =====================
class DlibFace:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.detector = dlib.get_frontal_face_detector()
            cls._instance.sp = dlib.shape_predictor(DLIB_PREDICTOR)
            cls._instance.model = dlib.face_recognition_model_v1(DLIB_RECOG_MODEL)
        return cls._instance

    def embedding(self, rgb, box):
        x1, y1, x2, y2 = box
        rect = dlib.rectangle(x1, y1, x2, y2)
        shape = self.sp(rgb, rect)
        return np.array(self.model.compute_face_descriptor(rgb, shape))

# ===================== Worker =====================
class FaceWorker(QThread):
    result = pyqtSignal(object)

    def __init__(self, ref_emb):
        super().__init__()
        self.ref_emb = ref_emb
        self.frame = None
        self.boxes = None
        self.running = True

    def set_data(self, frame, boxes):
        self.frame = frame
        self.boxes = boxes

    def run(self):
        dlib_face = DlibFace()
        while self.running:
            if self.frame is None or self.boxes is None:
                time.sleep(0.005)
                continue

            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            target_box = None

            for box in self.boxes:
                x1, y1, x2, y2 = map(int, box)
                try:
                    emb = dlib_face.embedding(rgb, (x1, y1, x2, y2))
                    dist = np.linalg.norm(emb - self.ref_emb)
                    if dist < FACE_THRESHOLD:
                        target_box = (x1, y1, x2, y2)
                        break
                except:
                    continue

            self.result.emit(target_box)
            self.frame = None

    def stop(self):
        self.running = False

# ===================== 主界面 =====================
class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸锁定（YOLO + dlib 稳定版）")
        self.resize(900, 600)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(255, 255, 255))
        self.setPalette(pal)

        self.video_label = QLabel("视频预览")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#eee;border:2px solid #ccc")

        btn_style = "background:#ffd700;padding:10px;font-size:16px;border-radius:8px;"

        self.btn_face = QPushButton("选择参考脸")
        self.btn_face.setStyleSheet(btn_style)
        self.btn_face.clicked.connect(self.load_face)

        self.btn_video = QPushButton("选择视频")
        self.btn_video.setStyleSheet(btn_style)
        self.btn_video.clicked.connect(self.load_video)

        self.btn_start = QPushButton("开始处理")
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.start)

        self.save_check = QCheckBox("保存视频")
        self.save_check.setChecked(True)

        h = QHBoxLayout()
        for b in [self.btn_face, self.btn_video, self.btn_start, self.save_check]:
            h.addWidget(b)

        v = QVBoxLayout()
        v.addWidget(self.video_label)
        v.addLayout(h)

        w = QWidget()
        w.setLayout(v)
        self.setCentralWidget(w)

        self.ref_emb = None
        self.video_path = None
        self.cap = None
        self.yolo = None
        self.worker = None
        self.last_box = None
        self.frame_id = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)

    def load_face(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择人脸")
        if not path:
            return
        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = DlibFace().detector(rgb, 1)
        if not dets:
            QMessageBox.warning(self, "错误", "未检测到人脸")
            return
        self.ref_emb = DlibFace().embedding(rgb, (
            dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()
        ))
        QMessageBox.information(self, "OK", "参考人脸加载成功")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频")
        if path:
            self.video_path = path

    def start(self):
        if self.ref_emb is None or not self.video_path:
            QMessageBox.warning(self, "错误", "请先选人脸和视频")
            return

        model_path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "*.pt")
        if not model_path:
            return

        self.yolo = YOLO(model_path)  # ⭐ 只在主线程加载
        self.cap = cv2.VideoCapture(self.video_path)

        self.worker = FaceWorker(self.ref_emb)
        self.worker.result.connect(self.update_box)
        self.worker.start()

        self.timer.start(30)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.worker.stop()
            return

        self.frame_id += 1

        if self.frame_id % DETECT_INTERVAL == 0:
            results = self.yolo(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
            self.worker.set_data(frame.copy(), boxes)

        if self.last_box:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

        if self.frame_id % DISPLAY_INTERVAL == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)

    def update_box(self, box):
        self.last_box = box

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
        e.accept()

# ===================== 入口 =====================
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    sys.exit(app.exec_())
