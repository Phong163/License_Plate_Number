import cv2
import matplotlib.pyplot as plt
import torch
from .detection import get_textbox
from .utils import group_text_box, reformat_input
import numpy as np
from pathlib import Path
import os
import sys
from .craft import CRAFT
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from ultralytics.utils import LOGGER, TryExcept, ops, plt_settings, threaded
from PIL import Image, ImageDraw, ImageFont

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False):
    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net

def detect(img,model, min_size=20, text_threshold=0.7, low_text=0.4,
           link_threshold=0.4, canvas_size=2560, mag_ratio=1.,
           slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
           width_ths=0.5, add_margin=0.1, optimal_num_chars=None,
           threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0):
    model_train=model
    img, img_cv_grey = reformat_input(img)
    text_box_list = get_textbox(model_train,
                                img,
                                canvas_size=canvas_size,
                                mag_ratio=mag_ratio,
                                text_threshold=text_threshold,
                                link_threshold=link_threshold,
                                low_text=low_text,
                                poly=False,
                                device='cpu',
                                optimal_num_chars=optimal_num_chars,
                                threshold=threshold,
                                bbox_min_score=bbox_min_score,
                                bbox_min_size=bbox_min_size,
                                max_candidates=max_candidates)

    horizontal_list_agg, free_list_agg = [], []
    for text_box in text_box_list:
        horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                    ycenter_ths, height_ths,
                                                    width_ths, add_margin,
                                                    (optimal_num_chars is None))
        if min_size:
            horizontal_list = [i for i in horizontal_list if max(
                i[1] - i[0], i[3] - i[2]) > min_size]
            
        horizontal_list_agg.append(horizontal_list)
  

    return horizontal_list_agg
'''
def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    if not isinstance(xyxy, torch.Tensor):
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]): int(xyxy[0, 3]), int(xyxy[0, 0]): int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)
        f = str(increment_path(file).with_suffix(".jpg"))
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)
    return crop
def crop_and_save_image(image_path, output_folder):
    """
    Cắt ảnh dựa trên tọa độ hai góc và lưu ảnh cắt vào một thư mục với tên ảnh chứa số thứ tự và tọa độ cắt.

    Args:
        image_path (str): Đường dẫn đến ảnh gốc.
        output_folder (str): Thư mục để lưu ảnh cắt.
        x_min (int): Tọa độ x của góc trên bên trái.
        y_min (int): Tọa độ y của góc trên bên trái.
        x_max (int): Tọa độ x của góc dưới bên phải.
        y_max (int): Tọa độ y của góc dưới bên phải.
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return

    # Cắt ảnh
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Tạo thư mục nếu chưa tồn tại
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Lấy số thứ tự ảnh trong thư mục
    file_count = len(list(output_folder.glob("*.jpg"))) + 1

    # Tạo tên tệp mới
    output_filename = f"{file_count}.jpg"
    output_path = output_folder / output_filename

    # Lưu ảnh cắt
    cv2.imwrite(str(output_path), cropped_image)
    print(f"Cropped image saved to: {output_path}")
'''
if __name__ == "__main__":
    img_path = r'C:\Users\OS\Desktop\My_project\PlateNumberRecognize\test_image\1.jpg'
    trained_model = r"C:\Users\OS\Desktop\My_project\PlateNumberRecognize\Craft\craft_mlt_25k.pth"
    img = cv2.imread(img_path)
    model = get_detector(trained_model)
    horizontal_list = detect(img,model)
    horizontal_list = horizontal_list[0]
    print('horizontal:',horizontal_list)
    for bbox in horizontal_list:
        x_min, x_max, y_min, y_max = bbox
        print('x_min:',x_min)
                
               
            
  
