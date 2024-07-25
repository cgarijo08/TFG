import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_imgs(title, image, pos = None):
    if isinstance(title, list):
        cv2.namedWindow(title[0], cv2.WINDOW_NORMAL)
        cv2.namedWindow(title[1], cv2.WINDOW_NORMAL)
        cv2.moveWindow(title[0], 0,0)
        cv2.moveWindow(title[1], 1920, 0)
        cv2.resizeWindow(title[0], 1920, 1080)
        cv2.resizeWindow(title[1], 1620, 1050)
        cv2.imshow(title[0], image[0])
        cv2.imshow(title[1], image[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if pos == 0:
            place = 0
            size = (1900, 1000)
        else:
            place = 1920
            size = (1620, 1050)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, *size)
        cv2.moveWindow(title, place, 0)
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def is_white_image(img):
    if np.var(img[:,:,0]) < 40:
        return True
    return False

def apply_morf_transformation(type:str, kern, iterations, image):
    if type == 'erode':
        return cv2.erode(image, kern, iterations=iterations)
    elif type == 'dilation':
        return cv2.dilate(image, kern, iterations=iterations)
    elif type == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kern, iterations=iterations)
    elif type == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kern, iterations=iterations)
    else:
        raise ValueError(f"{type} operation is not supported. The available operations are: 'erode', 'dilation', 'opening' or 'closing'.")

# Function that returns a patch of the image
def get_patch_of_image(image, patch_center, patch_width, patch_height):
    x_dim=patch_width/2
    y_dim=patch_height/2
    if patch_center[1]+x_dim > image.shape[1] or patch_center[1]-x_dim < 0 or patch_center[0]+y_dim > image.shape[0] or patch_center[0]-y_dim < 0:
        raise ValueError('Patch boundaries out of the image')

    return image[int(patch_center[0]-y_dim):int(patch_center[0]+y_dim), int(patch_center[1]-x_dim):int(patch_center[1]+x_dim), :]

def draw_shapes_and_contours(contours, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
        cv2.drawContours(mask, [cnt], -1, 128, 10)
    return mask

def extract_number(filename):
    return int(filename.split('.')[0])
def tiff_from_number(number):
    return str(number)+".tiff"

def create_plot(features):
    fig, ax = plt.subplots(1,2, figsize=(15,10))

    data = [[],[]]
    for feature in features['instances'].values():
        data[0].append(float(feature["area"]))
        data[1].append(float(feature["circularity"]))
    ax[0].boxplot(data[0])
    ax[1].boxplot(data[1])
    # Data to display in the box
    textstr = '\n'.join((
        r'$n=%.2f$' % (features["n_instances"], ),
        r'$mean=%.2f$' % (features["mean_area"], ),
        r'$circ=%.2f$' % (features["mean_circularity"], )))

    # Properties of the box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place the text box in the upper left in axes coordinates
    ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=7,
            verticalalignment='top', bbox=props)
    fig.canvas.draw()
    plot_image = np.array(fig.canvas.renderer._renderer)
    # Get the screen resolution
    screen_res = 1600, 1200  # Adjust this to your screen resolution

    # Calculate the scale factor to fit the image to the screen
    scale_width = screen_res[0] / plot_image.shape[1]
    scale_height = screen_res[1] / plot_image.shape[0]
    scale = min(scale_width, scale_height)

    # Calculate the new dimensions
    window_width = int(plot_image.shape[1] * scale)
    window_height = int(plot_image.shape[0] * scale)

    # Resize the image
    resized_img = cv2.resize(plot_image, (window_width, window_height))

    # Convert the color format from RGB to BGR
    plot_image_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
    return plot_image_bgr

