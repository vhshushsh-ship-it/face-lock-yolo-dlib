# 🔐 YOLO + dlib 人脸锁定（绝不崩 · 稳定版）

基于 **YOLO + dlib + PyQt5** 的实时人脸锁定系统，专注于 **多线程稳定性设计**，避免 PyQt + 深度学习常见的崩溃问题，适合作为课程设计、工程示例与人脸识别学习项目。

> ✔ Windows / PyQt5 / GPU / exe 打包环境下实测稳定

---

## ✨ 项目特性

* 🚀 **YOLO 实时人脸检测**（仅在主线程加载与推理）
* 🧠 **dlib 人脸特征提取与匹配**（QThread 子线程执行）
* 🧵 **严格的多线程职责划分**，避免 UI 卡死与闪退
* 🖼️ **PyQt5 图形界面**，实时显示视频与锁定框
* ⏱️ **限帧 + 间隔检测机制**，降低算力与线程压力

---

## 🧱 项目结构

```text
face-lock-yolo-dlib/
├── main.py              # 主程序（YOLO + dlib + PyQt5）
├── requirements.txt     # Python 依赖
├── .gitignore           # Git 忽略规则
└── README.md            # 项目说明文档
```

---

## ⚙️ 运行环境

* Python 3.8 ~ 3.11（推荐 3.9+）
* Windows / Linux / macOS
* 可用摄像头或本地视频文件
* GPU（可选，YOLO 自动使用）

---

## 📦 安装依赖

建议使用 **虚拟环境**：

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

⚠️ **注意**：

* Windows 下 `dlib` 可能需要 CMake + Visual Studio Build Tools
* 推荐 Python 3.9 + 预编译 wheel 或 Conda 安装

---

## 📁 额外模型文件（需自行准备）

本项目 **不会上传以下模型文件**，请自行下载并放在程序同目录：

* `shape_predictor_68_face_landmarks.dat`
* `dlib_face_recognition_resnet_model_v1.dat`
* 任意 YOLO 人脸检测模型（`.pt`）

---

## ▶️ 使用方法

### 1️⃣ 启动程序

```bash
python main.py
```

### 2️⃣ 操作流程

1. 点击 **「选择参考脸」**，加载单人正脸照片
2. 点击 **「选择视频」**，加载待处理视频
3. 选择 **YOLO 模型文件（.pt）**
4. 点击 **「开始处理」**

程序将：

* 使用 YOLO 检测视频中的人脸
* 使用 dlib 提取 embedding
* 自动锁定与参考人脸最相似的目标

---

## 🧠 核心设计说明（稳定性关键）

本项目严格遵循以下 **工程级稳定性原则**：

1️⃣ **YOLO 模型仅在主线程加载与推理**
2️⃣ **QThread 子线程仅负责：**

* dlib embedding 提取
* 人脸特征距离计算
  3️⃣ **线程之间仅传递数据，不传模型对象**
  4️⃣ **UI 显示限帧（DISPLAY_INTERVAL）**，防止事件队列堆积
  5️⃣ **dlib 模型单例模式**，全局只初始化一次

这些设计可有效避免：

* PyQt 崩溃 / 闪退
* OpenCV 卡死
* 多线程死锁
* 显存与内存泄漏

---

## ⚙️ 关键参数说明

```python
FACE_THRESHOLD = 0.45      # 人脸相似度阈值（越小越严格）
DISPLAY_INTERVAL = 2       # UI 刷新帧间隔
DETECT_INTERVAL = 5        # 人脸匹配帧间隔
```

---

## 🚧 可扩展方向（TODO）

* [ ] 多人注册与人脸库管理
* [ ] 实时摄像头模式
* [ ] GPU / CPU 自动切换
* [ ] 配置文件（yaml / json）支持
* [ ] exe 打包发布

---

## 📜 声明

本项目仅用于 **学习与研究用途**，请勿用于任何侵犯隐私或违法场景。

---

## ⭐ 致谢

* Ultralytics YOLO
* dlib
* OpenCV
* PyQt5

如果这个项目对你有帮助，欢迎 ⭐ Star 支持一下！




