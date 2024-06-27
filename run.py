import time
import cv2
from interface import Ui_MainWindow
from yolo.detect2 import run
from recognize.lp_recognition import E2E
from recognize.lp_recognition2 import E2E2

from Craft import run_craft
import sys
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from yolo.models.common import DetectMultiBackend
from yolo.utils.torch_utils import select_device
import os

from pathlib import Path
FILE = Path(__file__).resolve()
print('FILE:',FILE)
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Thêm ROOT vào PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def crop(image, x_min, x_max, y_min, y_max):
    return image[y_min:y_max, x_min:x_max]

class Interface(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(MainWindow)
        self.pushButton.clicked.connect(self.Detect1)
        self.pushButton_2.clicked.connect(self.Detect2)
        self.pushButton_browse.clicked.connect(self.browse_image)  # Kết nối nút Browse

        # set font put text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_color = (0, 0, 255)
        # path weight
        yolo_weights = ROOT /'weights/last.pt' # path weight for yolo
        print('ROOT:',ROOT)
        self.device = select_device("")
        self.model = DetectMultiBackend(yolo_weights, device=self.device, dnn=False, data=ROOT / "data/coco128.yaml", fp16=False)

        craft_weight = ROOT/ "weights/craft_mlt_25k.pth"
        self.model_craft = run_craft.get_detector(craft_weight)
        
        self.char_weight = ROOT / 'weights/weight3.h5' # path weight for recognition

        self.model_recognize_1 = E2E(self.char_weight)
        self.model_recognize_2 = E2E2(self.char_weight)




    def get_img_path(self):
        return self.textEdit.toPlainText()

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(MainWindow, "Chọn tệp ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.textEdit.setText(file_path)

    def display_image(self, img):
        # Chuyển đổi hình ảnh từ OpenCV sang QImage
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Chuyển đổi QImage sang QPixmap và hiển thị trong QLabel
        pixmap = QPixmap.fromImage(q_img)
        self.label_3.setPixmap(pixmap)
        self.label_3.setScaledContents(True)

    def Detect1(self):
        start_time = time.time()
        self.label.setText("")
        img_path = self.get_img_path()
        
        # crop yolo
        im0, crop_yolo_boxes = run(img_path,self.model, self.device)

        for item in crop_yolo_boxes:
            
            crop_plate, xyxy = item
            #crop Craft
            horizontal_list = run_craft.detect(crop_plate, self.model_craft)
            horizontal_list = horizontal_list[0]
            combined_license_plate = []
            license_plates = []
            for i, bbox in enumerate(horizontal_list):
                x2_min, x2_max, y2_min, y2_max = bbox
                text_crop = crop(crop_plate, x2_min, x2_max, y2_min, y2_max)
                if text_crop.size == 0:
                    print(f"Empty text_crop at bbox: {bbox}")
                    continue
                candidates = self.model_recognize_1.segmentation(text_crop)
                license_plate = self.model_recognize_1.recognizeChar(candidates)
                if license_plate:
                    license_plates.append(license_plate)
                if len(horizontal_list)>1 and i==0:
                    license_plates.append('-')
            combined_license_plate = ''.join(license_plates)

            x_min, y_min, x_max, y_max = map(int, xyxy)
            cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
         
            if license_plates:
                text_x = x_min
                text_y = y_min - 10  # Văn bản nằm phía trên hình chữ nhật
                # Kiểm tra và điều chỉnh nếu tọa độ văn bản nằm ngoài hình ảnh
                if text_y < 0:
                    text_y = y_min + 10  # Nếu nằm ngoài, đặt nó bên dưới hình chữ nhật

                print('combined_license_plate:',combined_license_plate)
                cv2.putText(im0, combined_license_plate, (text_x, text_y), self.font, self.font_scale, self.text_color, self.font_thickness)
                current_text = self.label.text()
                new_text = current_text + "\n" + combined_license_plate if current_text else combined_license_plate
                self.label.setText(new_text)
                self.label.adjustSize()
            else:
                self.label.setText("No TEXT")
            self.display_image(im0)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Thời gian chạy: {elapsed_time:.2f} giây")


    def Detect2(self):
        start_time = time.time()
        self.label.setText("")
        img_path = self.get_img_path()
        #crop yolo
        im0, crop_yolo_boxes = run(img_path, self.model, self.device)
        
        for item in crop_yolo_boxes:
            crop_plate, xyxy = item
            license_plate = self.model_recognize_2.predict(crop_plate)
            x_min, y_min, x_max, y_max = map(int, xyxy)
            cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            if license_plate:
                text_x = x_min
                text_y = y_min - 10  # Văn bản nằm phía trên hình chữ nhật
                # Kiểm tra và điều chỉnh nếu tọa độ văn bản nằm ngoài hình ảnh
                if text_y < 0:
                    text_y = y_min + 10  # Nếu nằm ngoài, đặt nó bên dưới hình chữ nhật
                print('combined_license_plate:',license_plate)
                cv2.putText(im0, license_plate, (text_x, text_y), self.font, self.font_scale, self.text_color, self.font_thickness)
                current_text = self.label.text()
                new_text = current_text + "\n" + license_plate if current_text else license_plate
                self.label.setText(new_text)
                self.label.adjustSize()
            else:
                self.label.setText("No TEXT")
            self.display_image(im0)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Thời gian chạy: {elapsed_time:.2f} giây")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Interface()
    MainWindow.show()
    sys.exit(app.exec_())
