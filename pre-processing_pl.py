import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp
from utils.crop_patch import crop_patch
from utils.get_patch_features import (
    get_avg_colour, 
    get_mean_contour_area,
    get_mean_contour_circularity,
    get_circularity,
    get_patch_features
)
import json

def extract_number(filename):
    return int(filename.split('.')[0])

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
        cv2.drawContours(mask, [cnt], -1, 128, 5)
    return mask

def process_image(full_image_path, is_WSI, output = None, write_intermediate_imgs=False):
    # Reading image
    if full_image_path:
        path_to_full_image = full_image_path
    else:
        path_to_full_image='/home/gdem/Documents/Data/Pre-proj-images/1198/RGB_WS.png'
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
    patch_features = {}
    

    # Creating the patient directory inside 'data/'
    patient_path = 'data/'+patient
    os.makedirs(patient_path, exist_ok=True) 
    print(patient, tile)

    # Saving a compressed image for visualization 
    path_to_compressed_image=patient_path +'/compressed_image.jpg'
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
        patch_path = str(patch_center[0]) + '-' + str(patch_center[1]) + '_' + str(patch_width) + 'x' + str(patch_height)
        patch_path = patient_path + '/' + patch_path

        os.makedirs(patch_path, exist_ok=True)

        # Saving the image patch into the patch directory
        cv2.imwrite(patch_path+'/image_patch.png', patch_of_the_image)
    else:
        patch_of_the_image = full_image
    
    patch_path = patient

    # Apply Mean Shift algorithm to smooth areas in the image
    #shifted_image = cv2.pyrMeanShiftFiltering(patch_of_the_image, 20, 30)

    # First we convert the patch from BGR to Grayscale
    bw_patch_of_the_image = cv2.cvtColor(patch_of_the_image, cv2.COLOR_BGR2GRAY)

    # Dump the images processed with simple cv2.threshold method
    if write_intermediate_imgs:
        folder_path = patch_path+'/processed_images/threshold/'
        os.makedirs(folder_path, exist_ok=True)

    # Obtaining the thresholded image with otsu and triangle algorithms
    _, thr_image_otsu = cv2.threshold(bw_patch_of_the_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Dumping the new processed images
    if write_intermediate_imgs:
        cv2.imwrite(folder_path+'thr_image_otsu.png', thr_image_otsu)

    # ## Morphological transformation
    # Henceforth it is going to be used the image processed with the threshold method, using the threshold calculated with the OTSU algorithm.
    # In this section, morfological transformations are going to be applied to the image with the objective to remove the noise, so we get a clean photo.
    # 
    # It is recommended to have the thressholded image with the contours being in maxValue and background in 0, so that's why I inverted the last two images.

    

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    open_image_33_ell = apply_morf_transformation('opening', kern, iterations=1, image=thr_image_otsu)

    if write_intermediate_imgs:
        temp_folder_path = folder_path + 'opening/'
        cv2.imwrite(temp_folder_path + 'open_image_33_ell.png', open_image_33_ell)

    # It seems that the rectangular kernel can be a little aggresive to the image. It has erased some contours of the image, meanwhile, the noise difference isn't very remarkable
    # So, like the rectangular shape doesn't remove noise better than the ellipsed one, we're going to continue with the ellipsed one.

    

    # Let's now try a closing to see if we can fill out the gaps in the contours
    closed_image_ell = apply_morf_transformation('closing', kern=np.ones((4,4), np.uint8), iterations=2, image=open_image_33_ell)

    if write_intermediate_imgs:
        temp_folder_path = folder_path + 'closing/'
        cv2.imwrite(temp_folder_path + 'closed_image_ell_44.png', closed_image_ell)

    # ## Contour identification
    # In this section, the functions findContours() and drawContours() are going to be used and analyzed to see if they can provide relevant information for our work. The goal is to obtain an image with just the contours of the crystals' shape.
    # 
    # For the patient 1198 the best result was closed_image_ell_44.png. This picture is the result of having the original grayscale-image patch thresholded with the OTSU algorithm, applied an opening with an ellipsed 3x3 filter and a 2-iteration-closing with a 4x4 rectangular kernel.
    # 
    # We are going to skip for now the image after Canny algorithm because findContours() is a more powerful function that can provide more accurate information than the Canny Algorithm. Maybe the Canny algorithm can be analyzed in a parallel way from this method.
    if output is not None:
        folder_path = output
        os.makedirs(os.path.join(folder_path, patient), exist_ok=True)
    else:
        os.makedirs(patch_path + '/processed_images/contour_detection', exist_ok=True)
        folder_path = patch_path + '/processed_images/contour_detection/'

    image = closed_image_ell.copy()

    contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_BGR_tocopy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Important to transform the image from grayscale to BGR because "image" only has 1 layer. When doing 
                                                            # this transformation, we obtain "input_image" with 3 layer so then the contour can be drawn.
    contourned_image=cv2.drawContours(image_BGR_tocopy.copy(), contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)#, hierarchy=hierarchy, maxLevel=7)

    cv2.imwrite(os.path.join(folder_path, patient, 'contours_full.png'), contourned_image)


    biggest_contour = -1 # Biggest area calculated for the next section.
    biggest_idx = -1
    for idx, cnt in enumerate(contours):
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > biggest_contour:
            biggest_contour = cnt_area # Biggest area calculated for the next section.
            biggest_idx = idx

    # Plotting the alive contours
    blank_image = cv2.cvtColor(np.zeros(bw_patch_of_the_image.shape, np.uint8), cv2.COLOR_GRAY2BGR)


    # In opposite to the previous part, in this part the noise is going to be removed by looking for the contours hierarchy. This option may be a little unstable.
    # 
    # The procedure is the following:
    # 1) First, the biggest contour area and its index are obtained (in the previous section).
    # 2) Then, the first child contour is obtained by the hierarchy array.
    # 3) In a while loop, every contour in the same hierarchy than the previous one is appended to the child_contours array until hierarchy array returns -1.
    # 4) The father is appended.

    # Trying to remove the noise from contours hierarchy
    # For 1198, (13000, 8100)_1400x900 the biggest contour is 47. Let's find which contours are its childs
    child_contours = []
    first_child = hierarchy[0][biggest_idx][2] # We obtain the index of the first child ( hierarchy[0][i] = [next, previous, child, parent] )

    child_contours.append(contours[first_child]) # We append the first child to the array of child contours
    next_idx = hierarchy[0][first_child][0] # We obtain the idx of the next value. (hierarcy[0][first_child][1] gives -1 value)

    while(next_idx != -1): # Hierarchy doesn't have circularity. It has a first and a last contours for each hierarchy.
        child_contours.append(contours[next_idx]) 
        next_idx = hierarchy[0][next_idx][0]

    child_contours.append(contours[biggest_idx]) # We append the every contours' father.

    image_with_children_orig = cv2.drawContours(patch_of_the_image.copy(), child_contours, -1, (0,255,0), 2)
    image_with_children_blank = cv2.drawContours(blank_image.copy(), child_contours, -1, (0,255,0), 2)

    cv2.imwrite(os.path.join(folder_path, patient, 'contours_ch.png'), image_with_children_orig)
    cv2.imwrite(os.path.join(folder_path, patient, 'contours_ch_blank.png'), image_with_children_blank)

    # Filter the contours by area and circularity
    area_threshold = 2000
    circularity_threshold = 0.1
    filtered_contours = []
    for cnt in child_contours:
        if cv2.contourArea(cnt) > area_threshold and get_circularity(cnt) > circularity_threshold:
            filtered_contours.append(cnt)
    
    filtered_contours_image_orig = cv2.drawContours(patch_of_the_image.copy(), child_contours, -1, (0,255,0), 2)
    filtered_contours_image_blank = cv2.drawContours(blank_image.copy(), child_contours, -1, (0,255,0), 2)

    cv2.imwrite(os.path.join(folder_path, patient, 'filtered.png'), filtered_contours_image_orig)
    cv2.imwrite(os.path.join(folder_path, patient, 'filtered_blank.png'), filtered_contours_image_blank)

    #mean_1, mean_2 = get_mean_contour_area(filtered_contours)
    #patch_features['avg_cnt_area_children'] = {
    #    "mean_1" : mean_1,
    #    "mean_2" : mean_2
    #}
    #patch_features['avg_cnt_circ_children'] = get_mean_contour_circularity(child_contours)
    image_with_shapes_and_contours = draw_shapes_and_contours(filtered_contours, blank_image.copy())
    cv2.namedWindow(f'{patient} {tile} contours', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'{patient} {tile} original', cv2.WINDOW_NORMAL)
    cv2.moveWindow(f'{patient} {tile} contours', 0,0)
    cv2.moveWindow(f'{patient} {tile} original', 1920, 0)
    cv2.resizeWindow(f'{patient} {tile} contours', 1920, 1080)
    cv2.resizeWindow(f'{patient} {tile} original', 1620, 1050)

    cv2.imshow(f'{patient} {tile} contours', image_with_shapes_and_contours)
    cv2.imshow(f'{patient} {tile} original', full_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #patch_features = get_patch_features(filtered_contours, patch_of_the_image.shape[:2])


    H_mean, A_mean, B_mean = get_avg_colour(patch_of_the_image)
    patch_features["avg_colour"] = {
        "hue" : H_mean,
        "A" : A_mean,
        "B": B_mean
    }
    
    with open(os.path.join(folder_path, patient, 'patch_features.json'), 'w') as file:
        json.dump(patch_features, file, indent=4, ensure_ascii=False)

def run_processing(img_dir, are_tiles, output = None, is_img_file = False, patient_start = None):
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
                temp_images = sorted(os.listdir(temp_folder), key=extract_number)
                for temp in temp_images:
                    image_path = os.path.join(temp_folder, temp)
                    process_image(full_image_path=image_path, is_WSI=False, output=output)

    else:
        process_image(full_image_path=img_dir, is_WSI=True, output=output)
    

def main():
    parser = argparse.ArgumentParser(description='Script that applies the processes to the sample image to get the clean contours.')

    # Exclusive group because it is needed to pass either a single image or a directory, not both at the same time.
    exc_group = parser.add_mutually_exclusive_group()
    exc_group.add_argument('-f', '--im_file', type=str, help='Image to be processed.')
    exc_group.add_argument('-d', '--im_dir', type=str, help='Specify if a a batch of images is wanted to be processed instead of an image. This argument must be a directory which contains the patient photos organised in the following way: patient_num/RGB_WS.png or patient_num/data/RGB_WS.png')
    parser.add_argument('-o', '--output', type = str, help='Specify output folder.')
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
        output = '/home/gdem/Documents/Data/Processed_images'

    # Creating the processed-data directory
    os.makedirs('data', exist_ok=True)
    
    if images_dir:
        run_processing(images_dir, True, output, patient_start = start)
    else:
        run_processing(full_image_path, output, is_img_file=True)

if __name__ == '__main__':
    main()