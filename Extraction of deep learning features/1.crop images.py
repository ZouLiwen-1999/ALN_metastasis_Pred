import cv2
import os
img_path='------'  # Path of images
mask_path='------'  # Path of masks
roi_img_path='------'  # Path to save cut images
roi_mask_path='------'  # Path to save cut masks

if not os.path.exists(roi_img_path):
    os.makedirs(roi_img_path)
if not os.path.exists(roi_mask_path):
    os.makedirs(roi_mask_path)

files=[x for x in os.listdir(img_path) if x[-3:] == "png"]
N=len(files)
for file in files:
    print(file)
    if file[-5] == "1":
        img = cv2.imread(os.path.join(img_path,file),cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(os.path.join(img_path,file))
    print(img.shape)
    mask = cv2.imread(os.path.join(mask_path,file),cv2.IMREAD_GRAYSCALE)

    th, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    X_min=10000
    Y_min=10000
    X_max=0
    Y_max=0
    for bbox in bounding_boxes:
        [x_min, y_min, w, h] = bbox 
        x_max=x_min+w
        y_max=y_min+h
        if x_min<X_min:
            X_min=x_min
        if y_min<Y_min:
            Y_min=y_min
        if x_max>X_max:
            X_max=x_max
        if y_max>Y_max:
            Y_max=y_max

    roi_img=img[Y_min:Y_max,X_min:X_max]
    roi_mask=mask[Y_min:Y_max,X_min:X_max]
    cv2.imwrite(os.path.join(roi_img_path,file),roi_img)
    cv2.imwrite(os.path.join(roi_mask_path,file),roi_mask)  
