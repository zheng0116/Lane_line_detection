import argparse
import cv2
import numpy as np


class LaneDetection:
    def __init__(self,ksize=(5, 5),sigma=(0, 0),threshold1=100,
        threshold2=200,
        aperture_size=3,
        direction_point=None,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        min_line_len=200,
        max_line_gap=400,
        x1L=None,
        x2L=None,
        x1R=None,
        x2R=None,
    ):
        self.ksize = ksize
        self.sigma = sigma
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.direction_point = direction_point
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.x1L = x1L
        self.x2L = x2L
        self.x1R = x1R
        self.x2R = x2R

    def __call__(self, img):
        gauss = self._image_preprocess(img)
        edge = self._edge_canny(gauss)
        roi = self._roi_trapezoid(edge)
        lines = self._Hough_line_fitting(roi)
        line_img = self._lane_line_fitting(img, lines)
        res = self._weighted_img_lines(img, line_img)
        return res

    def _image_preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, self.ksize, self.sigma[0], self.sigma[1])
        return gauss

    def _edge_canny(self, img):
        edge = cv2.Canny(img, self.threshold1, self.threshold2, self.aperture_size)
        return edge

    def _roi_trapezoid(self, img):
        h, w = img.shape[:2]
        # 车方向的中心点
        if self.direction_point is None:
            left_top = [w//2, h//2]
            right_top = [w//2, h//2]
        else:
            left_top = self.direction_point
            right_top = self.direction_point
        left_down = [int(w * 0.1), h]
        right_down = [int(w * 0.9), h]
        self.roi_points = np.array([left_down, left_top, right_top, right_down])
        # 填充梯形区域
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.roi_points, 255)
        # 目标区域提取：逻辑与
        roi = cv2.bitwise_and(img, mask)

        return roi

    def _Hough_line_fitting(self, img):
        lines = cv2.HoughLinesP(
            img, self.rho, self.theta, self.threshold, np.array([]),
            minLineLength=self.min_line_len, maxLineGap=self.max_line_gap
        )
        return lines
    def _lane_line_fitting(self, img, lines, color=(0, 255, 0), thickness=8):
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        right_x = []
        right_y = []
        left_x = []
        left_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2-y1)/(x2-x1))
                if slope <= -0.2:
                    left_x.extend((x1, x2))
                    left_y.extend((y1, y2))

                elif slope >= 0.2:
                    right_x.extend((x1, x2))
                    right_y.extend((y1, y2))
        if left_x and left_y:
            left_fit = np.polyfit(left_x, left_y, 1)
            left_line = np.poly1d(left_fit)

            if not self.x1L:
                x1L = int(img.shape[1] * 0.1)
            y1L = int(left_line(x1L))
            if not self.x2L:
                x2L = int(img.shape[1] * 0.4)
            y2L = int(left_line(x2L))
            # cv2.line(line_img, (x1L, y1L), (x2L, y2L), color, thickness)
            cv2.line(line_img, (x1L, y1L), (x2L, y2L), (255, 0, 0), thickness)  # Blue color for left lane
        if right_x and right_y:
            right_fit = np.polyfit(right_x, right_y, 1)
            right_line = np.poly1d(right_fit)
            if not self.x1R:
                x1R = int(img.shape[1] * 0.6)
            y1R = int(right_line(x1R))
            if not self.x2R:
                x2R = int(img.shape[1] * 0.9)
            y2R = int(right_line(x2R))
            cv2.line(line_img, (x1R, y1R), (x2R, y2R), (0, 0, 255), thickness)
        return line_img
    
    def _weighted_img_lines(self, img, line_img, α=1, β=1, λ=0.):
        res = cv2.addWeighted(img, α, line_img, β, λ)
        return res
    
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_path", type=str, default="", help="Input path of image.")
    parser.add_argument("-o", "--output_path", type=str, default="", help="Ouput path of image.")
    return parser.parse_args()

def main():
    args = parse_args()
    lanedetection = LaneDetection()
    if args.input_path.endswith('.jpg'):
        img = cv2.imread(args.input_path, 1)
        res = lanedetection(img)
        key = cv2.waitkey()
        x = np.hstack([img, res])
        cv2.imwrite(args.output_path, x),

    elif args.input_path.endswith('.mp4'):

        video_capture = cv2.VideoCapture(args.input_path)
        # video_capture = cv.VideoCapture(0)q
        if not video_capture.isOpened():
            print('Failed to open video!')
            exit()

        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(args.output_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), apiPreference=0, fps=fps, frameSize=size)
        total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[INFO] {} total frames in video".format(total))
        frameToStart = 0
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)
        while(True):
            ret, frame = video_capture.read()
            if not ret:
                break
            res = lanedetection(frame)
            out.write(res)
            cv2.imshow("video", res)
            # 键盘控制视频
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        video_capture.release()
        out.release()
        cv2.destroyAllWindows()
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
