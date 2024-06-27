import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
from imutils import perspective
import imutils
import time

from .data_utils import order_points, convert2Square, draw_labels_and_boxes
#from src.lp_detection.detect import detectNumberPlate
from .char_classification.model import CNN_Model
from skimage.filters import threshold_local

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


class E2E(object):
    def __init__(self, weight):
        
        self.image = np.empty((28, 28, 1))
        #self.detectLP = detectNumberPlate(LP_DETECTION_CFG['classes_path'], LP_DETECTION_CFG['config_path'], LP_DETECTION_CFG['weight_path'])
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights(weight)
    
    # def predict(self, image):
    #     self.image = image
    #     self.segmentation(image)
    #     self.recognizeChar()
    #     license_plate = self.format()
    #     print('license_plate:', license_plate)
        
       

    def segmentation(self, LpRegion):
        candidates = []
        print('float(LpRegion.shape[0]):',float(LpRegion.shape[0]))
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)
        labels = measure.label(thresh, connectivity=2, background=0)
        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
             
                (x, y, w, h) = cv2.boundingRect(contour)
                
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(thresh.shape[0])
                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.5< heightRatio < 1.5:
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    candidates.append((square_candidate, (y, x)))
        return candidates
    def recognizeChar(self,candidates):
        candidates= candidates
        characters = []
        coordinates = []
        a=1
        if a==1:

            for char, coordinate in candidates:
                characters.append(char)
                coordinates.append(coordinate)
        print('len character:',len(characters))        
        characters = np.array(characters)
        # Ensure characters have the correct shape (batch_size, height, width, channels)
        if len(characters) != 0:
            characters = characters.reshape((characters.shape[0], 28, 28, 1))

            result = self.recogChar.predict_on_batch(characters)
            result_idx = np.argmax(result, axis=1)

            candidates = []
            for i in range(len(result_idx)):
                if result_idx[i] == 31:    # if is background or noise, ignore it
                    continue
                candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))
            def take_second(s):
                return s[1]
            #
            first_line = []
            for candidate, coordinate in candidates:
                    first_line.append((candidate, coordinate[1]))
            first_line = sorted(first_line, key=take_second)

            license_plate = "".join([str(ele[0]) for ele in first_line])

            return license_plate
        else:
            print('No character is recognized')
    


