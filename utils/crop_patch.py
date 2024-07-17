import cv2
import numpy as np

def change_coords(coords):
    patch_center = ((int((coords[1][1] + coords[0][1]) / 2)), int((coords[1][0] + coords[0][0])/2))
    patch_height = int(abs(coords[1][1] - coords[0][1]))
    patch_width = int(abs(coords[1][0] - coords[0][0]))
    return patch_center, patch_width, patch_height

drawing = False
ix, iy = -1, -1
patch_coords = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, img_copy, patch_coords

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = cv2.rectangle(img.copy(), (ix, iy), (x, y), (0, 255, 0), int(img.shape[0]/200))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img.copy(), (ix, iy), (x, y), (0, 255, 0), 2)
        patch_coords = [(ix, iy), (x, y)]

def crop_patch(image):
    # Global variables
    # Resize the image to fit the screen
    global img
    img = image
    screen_res = 1920, 1080
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    # Create a resizable window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)

    global img_copy
    img_copy = img.copy()

    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', img_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

    if patch_coords:
        print(f"Selected patch coordinates: Top-left: {patch_coords[0]}, Bottom-right: {patch_coords[1]}")
    else:
        print("No patch was selected.")
    return change_coords(patch_coords)