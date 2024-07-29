import cv2
import os
from PIL import Image as PILImage
import numpy as np
import argparse
from tqdm import tqdm


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
    if patch_center[0]+x_dim > image.shape[1] or patch_center[0]-x_dim < 0 or patch_center[1]+y_dim > image.shape[0] or patch_center[1]-y_dim < 0:
        raise ValueError('Patch boundaries out of the image')

    return image[int(patch_center[0]-y_dim):int(patch_center[0]+y_dim), int(patch_center[1]-x_dim):int(patch_center[1]+x_dim), :]

def process_image(full_image_path, write_intermediate_imgs=False):
    # Reading image
    if full_image_path:
        path_to_full_image = full_image_path
    else:
        path_to_full_image='/home/gdem/Documents/Data/Pre-proj-images/1198/RGB_WS.png'
    if path_to_full_image.split('/')[-2] == 'data':
        patient = path_to_full_image.split('/')[-3]
    else:
        patient = path_to_full_image.split('/')[-2]
    if not os.path.isfile(path_to_full_image):
        raise FileNotFoundError('The path you introduced doesn\'t refer to any existing image. Please, make sure you introduce the correct path')
    full_image=cv2.imread(path_to_full_image)

    # Creating the patient directory inside 'data/'
    patient_path = 'data/'+patient
    os.makedirs(patient_path, exist_ok=True) 
    print(patient)

    # Saving a compressed image for visualization 
    path_to_compressed_image=patient_path +'/compressed_image.jpg'
    compressed_image=cv2.resize(full_image, (1920, 1080))
    compression_parms=[cv2.IMWRITE_JPEG_QUALITY, 60]

    cv2.imwrite(path_to_compressed_image, compressed_image, compression_parms)

    # Getting a patch of the image
    patch_center = (13000, 8100) # (Y, X)
    patch_width = 1400
    patch_height = 900

    patch_of_the_image = get_patch_of_image(full_image, patch_center=patch_center, patch_width=patch_width, patch_height=patch_height)

    # Creating directory of the patch
    patch_path = str(patch_center[0]) + '-' + str(patch_center[1]) + '_' + str(patch_width) + 'x' + str(patch_height)
    patch_path = patient_path + '/' + patch_path

    os.makedirs(patch_path, exist_ok=True)

    # Saving the image patch into the patch directory
    cv2.imwrite(patch_path+'/image_patch.png', patch_of_the_image)


    # First we convert the patch from BGR to Grayscale
    bw_patch_of_the_image = cv2.cvtColor(patch_of_the_image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thr_image_bin125 = cv2.threshold(bw_patch_of_the_image, 125, 255, cv2.THRESH_BINARY)
    _, thr_image_bin70 = cv2.threshold(bw_patch_of_the_image, 70, 255, cv2.THRESH_BINARY)
    _, thr_image_bin150 = cv2.threshold(bw_patch_of_the_image, 150, 255, cv2.THRESH_BINARY)
    _, thr_image_t0 = cv2.threshold(bw_patch_of_the_image, 125, 255, cv2.THRESH_TOZERO)


    # Dump the images processed with simple cv2.threshold method
    
    os.makedirs(patch_path+'/processed_images', exist_ok=True)

    os.makedirs(patch_path+'/processed_images/threshold', exist_ok=True)

    folder_path = patch_path+'/processed_images/threshold/'

    cv2.imwrite(folder_path+'thr_image_bin125.png', thr_image_bin125)
    cv2.imwrite(folder_path+'thr_image_bin70.png', thr_image_bin70)
    cv2.imwrite(folder_path+'thr_image_bin150.png', thr_image_bin150)
    cv2.imwrite(folder_path+'thr_image_t0.png', thr_image_t0)

    # Obtaining the thresholded image with otsu and triangle algorithms
    _, thr_image_otsu = cv2.threshold(bw_patch_of_the_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, thr_image_triangle = cv2.threshold(bw_patch_of_the_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_TRIANGLE)

    # Dumping the new processed images
    cv2.imwrite(folder_path+'thr_image_otsu.png', thr_image_otsu)
    cv2.imwrite(folder_path+'thr_image_triangle.png', thr_image_triangle)


    # ## Morphological transformation
    # Henceforth it is going to be used the image processed with the threshold method, using the threshold calculated with the OTSU algorithm.
    # In this section, morfological transformations are going to be applied to the image with the objective to remove the noise, so we get a clean photo.
    # 
    # It is recommended to have the thressholded image with the contours being in maxValue and background in 0, so that's why I inverted the last two images.

    # We create the folder where we are going to dump the images to which has been applied morphological transformation
    os.makedirs(patch_path+'/processed_images/morph_transformation', exist_ok=True)

    os.makedirs(patch_path+'/processed_images/morph_transformation/dilation', exist_ok=True)

    os.makedirs(patch_path+'/processed_images/morph_transformation/erotion', exist_ok=True)

    os.makedirs(patch_path+'/processed_images/morph_transformation/opening', exist_ok=True)

    os.makedirs(patch_path+'/processed_images/morph_transformation/closing', exist_ok=True)

    folder_path = patch_path+'/processed_images/morph_transformation/'



    temp_folder_path = folder_path + 'dilation/'

    # Dilation
    dilated_image_33 = apply_morf_transformation('dilation', np.ones((3,3), np.uint8), 1, thr_image_otsu)
    dilated_image_55 = apply_morf_transformation('dilation', np.ones((5,5), np.uint8), 1, thr_image_otsu)
    dilated_image_77 = apply_morf_transformation('dilation', np.ones((7,7), np.uint8), 1, thr_image_otsu)

    cv2.imwrite(temp_folder_path + 'dilated_image_33.png', dilated_image_33)
    cv2.imwrite(temp_folder_path + 'dilated_image_55.png', dilated_image_55)
    cv2.imwrite(temp_folder_path + 'dilated_image_77.png', dilated_image_77)

    temp_folder_path = folder_path + 'erotion/'

    # Eroding the image
    eroded_image_33 = apply_morf_transformation('erode', np.ones((3,3), np.uint8), 1, thr_image_otsu)
    eroded_image_55 = apply_morf_transformation('erode', np.ones((5,5), np.uint8), 1, thr_image_otsu)
    eroded_image_77 = apply_morf_transformation('erode', np.ones((7,7), np.uint8), 1, thr_image_otsu)

    cv2.imwrite(temp_folder_path + 'eroded_image_33.png', eroded_image_33)
    cv2.imwrite(temp_folder_path + 'eroded_image_55.png', eroded_image_55)
    cv2.imwrite(temp_folder_path + 'eroded_image_77.png', eroded_image_77)


    # It seems that the erode function may have removed some of the noise and for a kernel of (3,3) we obtained the best result
    # Let's try with a kernel of (4,4) and with another structural shapes with cv2.getStructuringElement

    kern = np.ones((4,4), np.uint8)
    eroded_image_44=apply_morf_transformation('erode', kern=kern, iterations=1, image=thr_image_otsu)

    cv2.imwrite(temp_folder_path + 'eroded_image_44.png', eroded_image_44)

    # It seems that some borders have been removed so maybe it isn't the optimal way to remove the noise because we have lost important information

    # Let's try with a (3,3) and (4,4) ellipse
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3,3))
    eroded_image_33_ellipse = apply_morf_transformation('erode', kern, iterations=1, image=thr_image_otsu)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(4,4))
    eroded_image_44_ellipse = apply_morf_transformation('erode', kern, iterations=1, image=thr_image_otsu)

    cv2.imwrite(temp_folder_path + 'eroded_image_33_ellipse.png', eroded_image_33_ellipse)
    cv2.imwrite(temp_folder_path + 'eroded_image_44_ellipse.png', eroded_image_44_ellipse)

    # (4,4) kernel is to aggresive so let's go back to (3,3).

    temp_folder_path = folder_path + 'opening/'

    # Let's try an opening with a rectangular kernel and ellipse kernel of size (3,3)

    kern = np.ones((3,3), np.uint8)
    open_image_33 = apply_morf_transformation('opening', kern, iterations=1, image=thr_image_otsu)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    open_image_33_ell = apply_morf_transformation('opening', kern, iterations=1, image=thr_image_otsu)

    cv2.imwrite(temp_folder_path + 'open_image_33.png', open_image_33)
    cv2.imwrite(temp_folder_path + 'open_image_33_ell.png', open_image_33_ell)

    # It seems that the rectangular kernel can be a little aggresive to the image. It has erased some contours of the image, meanwhile, the noise difference isn't very remarkable
    # So, like the rectangular shape doesn't remove noise better than the ellipsed one, we're going to continue with the ellipsed one.

    temp_folder_path = folder_path + 'closing/'

    # Let's now try a closing to see if we can fill out the gaps in the contours

    closed_image = apply_morf_transformation('closing', kern=np.ones((4,4), np.uint8), iterations=2, image=open_image_33)

    closed_image_ell = apply_morf_transformation('closing', kern=np.ones((4,4), np.uint8), iterations=2, image=open_image_33_ell)

    cv2.imwrite(temp_folder_path + 'closed_image_44.png', closed_image)
    cv2.imwrite(temp_folder_path + 'closed_image_ell_44.png', closed_image_ell)

    # ## Contour identification
    # In this section, the functions findContours() and drawContours() are going to be used and analyzed to see if they can provide relevant information for our work. The goal is to obtain an image with just the contours of the crystals' shape.
    # 
    # For the patient 1198 the best result was closed_image_ell_44.png. This picture is the result of having the original grayscale-image patch thresholded with the OTSU algorithm, applied an opening with an ellipsed 3x3 filter and a 2-iteration-closing with a 4x4 rectangular kernel.
    # 
    # We are going to skip for now the image after Canny algorithm because findContours() is a more powerful function that can provide more accurate information than the Canny Algorithm. Maybe the Canny algorithm can be analyzed in a parallel way from this method.

    os.makedirs(patch_path + '/processed_images/contour_detection', exist_ok=True)
    folder_path = patch_path + '/processed_images/contour_detection/'

    image = closed_image_ell.copy()

    contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_BGR_tocopy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Important to transform the image from grayscale to BGR because "image" only has 1 layer. When doing 
                                                            # this transformation, we obtain "input_image" with 3 layer so then the contour can be drawn.
    contourned_image=cv2.drawContours(image_BGR_tocopy.copy(), contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)#, hierarchy=hierarchy, maxLevel=7)

    cv2.imwrite(folder_path + 'contourned_image_full.png', contourned_image)


    # In this part I'm going to try removing the noise from the image by measuring every contour area. We set a threshold to remove every area below this threshold.
    area_threshold = 1000
    contours_higher_thr = []
    biggest_contour = -1 # Biggest area calculated for the next section.
    biggest_idx = -1
    for idx, cnt in enumerate(contours):
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > area_threshold:
            contours_higher_thr.append(cnt)
        if cnt_area > biggest_contour:
            biggest_contour = cnt_area # Biggest area calculated for the next section.
            biggest_idx = idx

    # Plotting the alive contours
    blank_image = cv2.cvtColor(np.zeros(bw_patch_of_the_image.shape, np.uint8), cv2.COLOR_GRAY2BGR)

    alive_contours_over_orig = cv2.drawContours(patch_of_the_image.copy(), contours_higher_thr, -1, (0,0,255), 2)
    alive_contours_over_blank = cv2.drawContours(blank_image.copy(), contours_higher_thr, -1, (0,0,255), 2)

    cv2.imwrite(folder_path+'area_alive_contours_over_orig_'+str(area_threshold)+'.png', alive_contours_over_orig)
    cv2.imwrite(folder_path+'area_alive_contours_over_blank_'+str(area_threshold)+'.png', alive_contours_over_blank)

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

    if write_intermediate_imgs:
        cv2.imwrite(folder_path+'image_with_children_orig.png', image_with_children_orig)
        cv2.imwrite(folder_path+'image_with_children_blank.png', image_with_children_blank)

def main():
    parser = argparse.ArgumentParser(description='Script that applies the processes to the sample image to get the clean contours.')

    # Exclusive group because it is needed to pass either a single image or a directory, not both at the same time.
    exc_group = parser.add_mutually_exclusive_group()
    exc_group.add_argument('-f', '--im_file', type=str, help='Image to be processed.')
    exc_group.add_argument('-d', '--im_dir', type=str, help='Specify if a a batch of images is wanted to be processed instead of an image. This argument must be a directory which contains the patient photos organised in the following way: patient_num/RGB_WS.png or patient_num/data/RGB_WS.png')
    args = parser.parse_args()
    images_dir = args.im_dir
    full_image_path = args.im_file

    # Creating the processed-data directory
    os.makedirs('data', exist_ok=True)
    
    if images_dir:
        parent_folder_images = os.listdir(images_dir)
        patient_images = []
        for p_dir in parent_folder_images:
            if 'RGB_WS.png' in os.listdir(os.path.join(images_dir, p_dir)):
                patient_image_path = os.path.join(images_dir, p_dir, 'RGB_WS.png')
                patient_images.append(patient_image_path)
            elif 'data' in os.listdir(os.path.join(images_dir, p_dir)):
                patient_image_path = os.path.join(images_dir, p_dir, 'data', 'RGB_WS.png')
                patient_images.append(patient_image_path)
        
        for im in tqdm(patient_images):
            process_image(full_image_path=im)
    else:
        process_image(full_image_path=full_image_path)

if __name__ == '__main__':
    main()