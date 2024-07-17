import argparse
import os
import cv2
from multiprocessing import Process

def get_patient(image):
    if image.split('/')[-2] == 'data':
        patient = image.split('/')[-3]
    else:
        patient = image.split('/')[-2]
    if not os.path.isfile(image):
        raise FileNotFoundError('The path you introduced doesn\'t refer to any existing image. Please, make sure you introduce the correct path')
    return patient

def visualize_single_image(full_image, patient):

    cv2.namedWindow(patient, cv2.WINDOW_NORMAL)
    cv2.imshow(patient, full_image)
    cv2.waitKey()
    cv2.destroyWindow(patient)

def visualize_multiple_images(images):
    next_image = []
    for idx, image_path in enumerate(images):
        
        if len(next_image) == 0:
            actual_image = cv2.imread(image_path)
        else:
            actual_image = next_image
        patient = get_patient(image_path)
        p = Process(target=visualize_single_image, args=(actual_image, patient))
        p.start()
        if idx+1 < len(images):
            try:
                next_image = cv2.imread(images[idx+1])
            except:
                print(f"Couldn't read image {get_patient(images[idx+1])}")
                pass
        p.join()

def main():
    parser = argparse.ArgumentParser(description='Script that applies the processes to the sample image to get the clean contours.')

    # Exclusive group because it is needed to pass either a single image or a directory, not both at the same time.
    exc_group = parser.add_mutually_exclusive_group()
    exc_group.add_argument('-f', '--im_file', type=str, help='Image to be processed.')
    exc_group.add_argument('-d', '--im_dir', type=str, help='Specify if a a batch of images is wanted to be processed instead of an image. This argument must be a directory which contains the patient photos organised in the following way: patient_num/RGB_WS.png or patient_num/data/RGB_WS.png')
    parser.add_argument('--start', type=str, help='Name of the first patient. Previous will be omitted.')
    args = parser.parse_args()
    images_dir = args.im_dir
    full_image_path = args.im_file
    if args.start:
        start = args.start
    else:
        start = None
    
    if images_dir:
        parent_folder_images = sorted(os.listdir(images_dir))
        if start is not None:
            parent_folder_images = parent_folder_images[parent_folder_images.index(start):]
        patient_images = []
        for p_dir in parent_folder_images:
            if 'RGB_WS.png' in os.listdir(os.path.join(images_dir, p_dir)):
                patient_image_path = os.path.join(images_dir, p_dir, 'RGB_WS.png')
                patient_images.append(patient_image_path)
            elif 'data' in os.listdir(os.path.join(images_dir, p_dir)):
                patient_image_path = os.path.join(images_dir, p_dir, 'data', 'RGB_WS.png')
                patient_images.append(patient_image_path)
        visualize_multiple_images(patient_images)
    else:
        patient = get_patient(full_image_path)
        image = cv2.imread(full_image_path)
        visualize_single_image(full_image=image, patient=patient)

if __name__ == '__main__':
    main()