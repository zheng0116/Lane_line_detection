import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
from aroad import LaneDetection
class AroadApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Line Detection System")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            AroadApp {
                background-image: url('1.jpg');
                background-repeat: no-repeat;
                background-position: center;
            }
        """)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.layout.addWidget(self.image_label)
        self.image_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #000000; /* Black border */
                        background-color: transparent; /* Transparent background */
                    }
                """)

        button_layout = QVBoxLayout()

        self.select_file_button = QPushButton("选择文件", self)
        self.set_button_style(self.select_file_button)
        self.select_file_button.clicked.connect(self.select_file)
        button_layout.addWidget(self.select_file_button)

        self.detect_button = QPushButton("检测车道线", self)
        self.set_button_style(self.detect_button)
        self.detect_button.clicked.connect(self.detect_lane)
        button_layout.addWidget(self.detect_button)

        self.start_button = QPushButton("开始视频", self)
        self.set_button_style(self.start_button)
        self.start_button.clicked.connect(self.start_video)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止视频", self)
        self.set_button_style(self.stop_button)
        self.stop_button.clicked.connect(self.stop_video)
        button_layout.addWidget(self.stop_button)
        self.save_button = QPushButton("保存结果", self)
        self.set_button_style(self.save_button)
        self.save_button.clicked.connect(self.save_result)
        button_layout.addWidget(self.save_button)

        self.layout.addLayout(button_layout)

        self.file_path = ""
        self.lanedetection = LaneDetection()

        self.video_capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)
        self.is_video_playing = False

    def set_button_style(self, button):
        button.setStyleSheet("""
            QPushButton {
                background: linear-gradient(to bottom, #3498db, transparent), radial-gradient(circle, #e74c3c, transparent)
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }

            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

    def detect_lane(self):
        if self.file_path:
            img = cv2.imread(self.file_path)
            res = self.lanedetection(img)
            self.update_image_label(res)


    def start_video(self):
        if self.file_path and self.file_path.endswith('.mp4'):
            self.video_capture = cv2.VideoCapture(self.file_path)
            self.is_video_playing = True
            self.timer.start(30)  # Update with desired frame rate



    def save_result(self):
        if self.file_path:
            img = cv2.imread(self.file_path)
            res = self.lanedetection(img)
            output_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "Image Files (*.jpg);;Video Files (*.mp4)")
            if output_path:
                if output_path.endswith('.jpg'):
                    cv2.imwrite(output_path, res)
                elif output_path.endswith('.mp4'):
                    fps = 30  # 设置视频帧率
                    size = (res.shape[1], res.shape[0])
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                    out.write(res)
                    out.release()



    def stop_video(self):
        if self.is_video_playing:
            self.is_video_playing = False
            self.timer.stop()
            if self.video_capture:
                self.video_capture.release()

    def update_video_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            res = self.lanedetection(frame)
            self.update_image_label(res)



    def update_image_label(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)



    def select_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image/Video Files (*.jpg *.png *.mp4)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_path, _ = file_dialog.getOpenFileName(self, "选择文件", "", "Image/Video Files (*.jpg *.png *.mp4)", options=options)


        if file_path:
            self.file_path = file_path
            self.image_label.clear()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    aroad_app = AroadApp()
    aroad_app.show()
    sys.exit(app.exec_())
