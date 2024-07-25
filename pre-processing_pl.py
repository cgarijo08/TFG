import cv2
import os
from sys import exit
import traceback
from shutil import rmtree
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp
from utils.crop_patch import crop_patch
from utils.get_patch_features import (
    get_avg_colour,
    get_circularity,
    get_patch_features
)
import json
import logging
import time
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Exception types
DIRECTORY_NOT_EXISTS = 0
CANT_LOAD_IMAGE = 1
WHITE_IMAGE = 2
NO_CONTOURS = 3

# Constants
FIRST_AREA_THRESHOLD = 2000
SECOND_AREA_THRESHOLD = 6000
CIRCULARITY_THRESHOLD = 0.1

# Delete previous logs
for log in os.listdir('logs'):
    if os.path.isdir(os.path.join('logs', log)):
        rmtree(os.path.join('logs', log))
    else:
        os.remove(os.path.join('logs', log))

def setup_logger(type_message, patient = None):
    if logger.hasHandlers():
        logger.handlers.clear()
    if type_message == DIRECTORY_NOT_EXISTS:
        file_handler = logging.FileHandler('./logs/directory_error.log', encoding='utf-8', mode='a')
    elif type_message == CANT_LOAD_IMAGE:
        file_handler = logging.FileHandler('./logs/load_image_error.log', encoding='utf-8', mode='a')
    elif type_message == WHITE_IMAGE:
        log_path = os.path.join('logs', 'white', patient)
        os.makedirs(log_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_path, 'white_image.log'), encoding='utf-8', mode='a')
    elif type_message == NO_CONTOURS:
        log_path = os.path.join('logs', 'no_contours', patient)
        os.makedirs(log_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_path, 'no_contours.log'), encoding='utf-8', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

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

def process_image(full_image_path, is_WSI, output = None, write_intermediate_imgs=False):
    start_process = time.time()
    # Reading image
    if full_image_path:
        path_to_full_image = full_image_path
    else:
        path_to_full_image='/home/gdem/Documents/Data/TFG/1198/RGB_WS.png'
    if path_to_full_image.split('/')[-2] == 'data':
        patient = path_to_full_image.split('/')[-3]
        tile = ''
    elif path_to_full_image.split('/')[-3] == 'data':
        patient = path_to_full_image.split('/')[-4]
        tile = path_to_full_image.split('/')[-1].split('.')[-2]
    else:
        patient = path_to_full_image.split('/')[-2]
        tile = ''
    if not os.path.isfile(path_to_full_image):
        raise FileNotFoundError('The path you introduced doesn\'t refer to any existing image. Please, make sure you introduce the correct path')
    full_image=cv2.imread(path_to_full_image)
    if is_white_image(full_image):
        setup_logger(WHITE_IMAGE, patient)
        logger.info(f"Tile: {tile} from patient: {patient} is a white image")
        cv2.namedWindow('H', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('H', 1900, 1000)
        cv2.imshow('H', full_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    patch_features = {}
    #visualize_imgs(f"{patient} {tile}", full_image, 0)

    # Creating the patient directory inside 'data/'
    patch_path = 'data/'+patient
    if tile != '':
        patch_path = os.path.join(patch_path, tile)
    os.makedirs(patch_path, exist_ok=True) 
    print(patient, tile)

    output_path = os.path.join(output, patient, tile)
    os.makedirs(output_path, exist_ok=True)

    # Saving a compressed image for visualization
    if write_intermediate_imgs:
        path_to_compressed_image=patch_path +'/compressed_image.jpg'
        compressed_image=cv2.resize(full_image, (1920, 1080))
        compression_parms=[cv2.IMWRITE_JPEG_QUALITY, 60]

        cv2.imwrite(path_to_compressed_image, compressed_image, compression_parms)

    if False:
        if False:
            # Getting a patch of the image
            patch_center = (13000, 8100) # (Y, X)
            patch_width = 1400
            patch_height = 900
        else:
            patch_center, patch_width, patch_height = crop_patch(full_image)

        patch_of_the_image = get_patch_of_image(full_image, patch_center=patch_center, patch_width=patch_width, patch_height=patch_height)

        # Creating directory of the patch
        patch_coords = str(patch_center[0]) + '-' + str(patch_center[1]) + '_' + str(patch_width) + 'x' + str(patch_height)
        patch_path = patch_path + '/' + patch_coords

        os.makedirs(patch_path, exist_ok=True)

        # Saving the image patch into the patch directory
        cv2.imwrite(patch_path+'/image_patch.png', patch_of_the_image)
    else:
        patch_of_the_image = full_image
    
    # Extracting Red channel because it has more presence in the images
    bw_patch_of_the_image = patch_of_the_image[:,:,2]

    # Dump the images processed with simple cv2.threshold method
    if write_intermediate_imgs:
        folder_path = os.path.join(patch_path, 'processed_images')
        temp_folder_path = os.path.join(folder_path, 'processed_images','threshold')
        os.makedirs(temp_folder_path, exist_ok=True)

    # Obtaining the thresholded image with an adaptive threshold
    thr_image_adaptive = cv2.adaptiveThreshold(bw_patch_of_the_image[:,:], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV , 51, 15)
    #visualize_imgs(f"thr {patient} {tile}", thr_image_adaptive, 0)
    #open_img = apply_morf_transformation('opening', cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1, thr_image_adaptive)
    #visualize_imgs('open', open_img, 0)
    
    contours, hierarchy = cv2.findContours(thr_image_adaptive, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    area_filtered_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > FIRST_AREA_THRESHOLD:# and get_circularity(cnt) > CIRCULARITY_THRESHOLD:
            area_filtered_contours.append(cnt)
    area_filtered_contours = sorted(area_filtered_contours, key=cv2.contourArea, reverse=True)
    contourned_image=cv2.drawContours(np.zeros((thr_image_adaptive.shape), dtype=np.uint8), contours=area_filtered_contours, contourIdx=-1, color=255, thickness=5)#, hierarchy=hierarchy, maxLevel=7)
    #visualize_imgs(f"First contours {patient} {tile}", contourned_image)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    close_image_33_ell = apply_morf_transformation('closing', kern, iterations=1, image=contourned_image)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    close_image_33_ell_2 = apply_morf_transformation('closing', kern, iterations=2, image=close_image_33_ell)
    invert_close_image = cv2.bitwise_not(close_image_33_ell_2)

    #visualize_imgs(f"Inv closed image {patient} {tile}", invert_close_image, 0)
    inv_second_contours, _ = cv2.findContours(invert_close_image, cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # Invertied
    isboundary = False
    inv_area_filtered_second_contours = []
    for idx, cnt in enumerate(inv_second_contours):
        isboundary = False
        for point in cnt:
            if any([coord == 0 for coord in point[0]]):
                isboundary = True
                break
            if point[0][0] == full_image.shape[1]-5:
                isboundary = True
                break
            if point[0][1] == full_image.shape[0]-5:
                isboundary = True
                break
        if not isboundary:
            if cv2.contourArea(cnt) > SECOND_AREA_THRESHOLD and get_circularity(cnt) > CIRCULARITY_THRESHOLD:
                inv_area_filtered_second_contours.append(cnt)
    inv_area_filtered_second_contours = sorted(inv_area_filtered_second_contours, key=cv2.contourArea, reverse=True)
    inv_shape_and_contourned_image = draw_shapes_and_contours(inv_area_filtered_second_contours, np.zeros((thr_image_adaptive.shape), dtype=np.uint8))
    orig_with_contours = cv2.drawContours(patch_of_the_image.copy(), inv_area_filtered_second_contours, -1, (0, 255, 0), 8)
         
    if not len(inv_area_filtered_second_contours) > 0:
        setup_logger(NO_CONTOURS, patient)
        logger.warning(f"Tile: {tile} from patient: {patient} has resulted in 0 contours.")
        return
    cv2.imwrite(os.path.join(output_path, 'shape_and_contours.png'), inv_shape_and_contourned_image)
    start = time.time()
    patch_features = get_patch_features(inv_area_filtered_second_contours, patch_of_the_image.shape[:2])
    H_mean, A_mean, B_mean = get_avg_colour(patch_of_the_image)
    patch_features["avg_colour"] = {
        "hue" : H_mean,
        "A" : A_mean,
        "B": B_mean
    }
    print(f'Features extracted in {time.time()-start}')
    
    #visualize_imgs([f"Orig with contours {patient} {tile}", f"Data"], [orig_with_contours, create_plot(patch_features)])
    #visualize_imgs(f"Orig with contours {patient} {tile}", orig_with_contours)
    with open(os.path.join(output_path, 'patch_features.json'), 'w') as file:
        json.dump(patch_features, file, indent=4, ensure_ascii=False)
    print(f"Whole process: {time.time()-start_process}\n")

def run_processing(img_dir, are_tiles, output = None, is_img_file = False, patient_start = None, tile_start = None, write_intermediate_imgs = False):
    if not is_img_file:
        parent_folder_images = os.listdir(img_dir)
        parent_folder_images = sorted(parent_folder_images)
        if not are_tiles:
            if patient_start is not None:
                parent_folder_images = parent_folder_images[parent_folder_images.index(patient_start):]
            patient_images = []
            for p_dir in parent_folder_images:
                if 'RGB_WS.png' in os.listdir(os.path.join(img_dir, p_dir)):
                    patient_image_path = os.path.join(img_dir, p_dir, 'RGB_WS.png')
                    patient_images.append((patient_image_path, True))
                elif 'data' in os.listdir(os.path.join(img_dir, p_dir)):
                    if 'RGB_WS.png' in os.listdir(os.path.join(img_dir, p_dir, 'data')):
                        patient_image_path = os.path.join(img_dir, p_dir, 'data', 'RGB_WS.png')
                        patient_images.append((patient_image_path, True))
                    elif 'RGB_PATCH.tiff' in os.listdir(os.path.join(img_dir, p_dir, 'data')):
                        patient_image_path = os.path.join(img_dir, p_dir, 'data', 'RGB_PATCH.tiff')
                        patient_images.append((patient_image_path, False))
            
            for im, is_WSI in tqdm(patient_images):
                process_image(full_image_path=im, is_WSI=is_WSI, output=output)
        else:
            if patient_start is not None:
                parent_folder_images = parent_folder_images[parent_folder_images.index(patient_start):]
            for patient in parent_folder_images:
                temp_folder = os.path.join(img_dir, patient, 'data', 'temp')
                if not os.path.exists(temp_folder):
                    setup_logger(DIRECTORY_NOT_EXISTS)
                    logger.error(f"{temp_folder} PATH DOESN'T EXIST.")
                    continue
                temp_images = sorted(os.listdir(temp_folder), key=extract_number)
                if tile_start is not None:
                    try:
                        temp_images = temp_images[temp_images.index(tiff_from_number(tile_start)):]
                    except:
                        print(traceback.format_exc())
                        continue
                for temp in temp_images:
                    image_path = os.path.join(temp_folder, temp)
                    try:
                        process_image(full_image_path=image_path, is_WSI=False, output=output, write_intermediate_imgs=write_intermediate_imgs)
                    except KeyboardInterrupt:
                        print(traceback.format_exc())
                        exit()
                    except:
                        setup_logger(CANT_LOAD_IMAGE)
                        logger.error(f"Patient: {patient}, tile: {temp} COULDN'T BE PROCESSED.\nError: {traceback.format_exc()}\n\n")
                        pass

    else:
        process_image(full_image_path=img_dir, is_WSI=True, output=output)
    

def main():
    parser = argparse.ArgumentParser(description='Script that applies the processes to the sample image to get the clean contours.')

    # Exclusive group because it is needed to pass either a single image or a directory, not both at the same time.
    exc_group = parser.add_mutually_exclusive_group()
    exc_group.add_argument('-f', '--im_file', type=str, help='Image to be processed.')
    exc_group.add_argument('-d', '--im_dir', type=str, help='Specify if a a batch of images is wanted to be processed instead of an image. This argument must be a directory which contains the patient photos organised in the following way: patient_num/RGB_WS.png or patient_num/data/RGB_WS.png')
    parser.add_argument('-o', '--output', type = str, help='Specify output folder.')
    parser.add_argument('-w', action='store_true', help='Write intermediate files in the project folder')
    parser.add_argument('--start', type=str, help='Start processing from this patient')
    args = parser.parse_args()
    images_dir = args.im_dir
    full_image_path = args.im_file
    output = args.output
    if args.start == None:
        start = None
    else:
        start = args.start
    if images_dir == None:
        images_dir = '/media/gdem/SSD/Carlos_data/DRY_SERUMS/patients/'
    if output == None:
        output = '/home/gdem/Documents/Data/Processed_images_v2'
    if args.w:
        write_intermediate_images = True
    else:
        write_intermediate_images = False

    # Creating the processed-data directory
    os.makedirs('data', exist_ok=True)
    
    if images_dir:
        run_processing(images_dir, True, output, patient_start = None, tile_start = None, write_intermediate_imgs=False) #FIS001
    else:
        run_processing(full_image_path, output, is_img_file=True)

if __name__ == '__main__':
    main()