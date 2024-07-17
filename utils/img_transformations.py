import cv2

def apply_gaussian(img, kern):
    pass

def maximize_contrast(img):
    mono_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_img = mono_img.max()
    min_img = mono_img.min()
    mid_value = (max_img - min_img) / 2
    mono_img = mono_img - mid_value
    alpha = 127 / mono_img.max()
    pass

