import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(image):
    light_orange =  np.array([1, 190, 150]) 
    dark_orange =  np.array([30, 255, 255]) 
    light_white =  np.array([60, 0, 200])   
    dark_white =  np.array([145, 150, 255]) 
    
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   
    mask_orange = cv2.inRange(hsv_img, light_orange, dark_orange)
    mask_white = cv2.inRange(hsv_img, light_white, dark_white)
    
    hsv_img[(mask_orange > 0) | (mask_white > 0)] = ([255, 255, 255])
    hsv_img[(mask_orange == 0) & (mask_white == 0)] = ([0, 0, 0])
    
    kernel_opening = np.ones((5,5),np.uint8)
    kernel_closing = np.ones((35,35),np.uint8)
    hsv_img = cv2.morphologyEx(hsv_img, cv2.MORPH_OPEN, kernel_opening)
    hsv_img = cv2.morphologyEx(hsv_img, cv2.MORPH_CLOSE, kernel_closing)
    return hsv_img[:,:,0]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
