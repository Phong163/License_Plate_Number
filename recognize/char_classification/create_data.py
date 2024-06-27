import os
import numpy as np
import cv2

# Hàm xử lý ảnh và lưu dữ liệu
def process_images(path, label_dict, output_file):
    data = []

    for fi in os.listdir(path):
        if fi in label_dict:
            label = label_dict[fi]
        else:
            raise ValueError(f"Không khớp file: {fi}")

        img_fi_path = os.listdir(os.path.join(path, fi))
        for img_path in img_fi_path:
            img = cv2.imread(os.path.join(path, fi, img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            data.append((img, label))

    data = np.array(data, dtype=object)  # Sử dụng dtype=object để lưu trữ các tuple
    np.save(output_file, data)

# Định nghĩa ánh xạ nhãn cho chữ số
digit_labels = {
    "0": 21, "1": 22, "2": 23, "3": 24, "4": 25, "5": 26, "6": 27,
    "7": 28, "8": 29, "9": 30, "BG": 31
}

# Xử lý và lưu trữ ảnh chữ số
process_images("./data/categorized/digits/", digit_labels, "./data/digits.npy")

# Định nghĩa ánh xạ nhãn cho chữ cái
alpha_labels = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, 
    "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "R": 13, "S": 14, 
    "T": 15, "U": 16, "V": 17, "X": 18, "Y": 19, "Z": 20
}

# Xử lý và lưu trữ ảnh chữ cái
process_images("./data/categorized/alphas/", alpha_labels, "./data/alphas.npy")
