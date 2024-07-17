import cv2
import numpy as np
from math import pi, copysign, log10
import mahotas

def get_avg_colour(patch):
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    H_mean = hsv_patch[:,:,0].mean()
    A_mean = lab_patch[:,:,1].mean()
    B_mean = lab_patch[:,:,2].mean()
    return H_mean, A_mean, B_mean
    
def get_mean_contour_area(contours):
    mean_1 = np.mean([cv2.contourArea(cnt) for cnt in contours])
    #mean_2 = np.sqrt(np.mean([cv2.contourArea(cnt)**2 for cnt in contours]))
    return mean_1#, mean_2

def get_circularity(contour):
    return ( 4*pi*cv2.contourArea(contour) ) / ( cv2.arcLength(contour, True)**2 ) 

def get_mean_contour_circularity(contours):
    return np.mean([( 4*pi*cv2.contourArea(cnt) ) / ( cv2.arcLength(cnt, True)**2 ) for cnt in contours])

def get_HU_moments(cnt):
    m = cv2.moments(cnt)
    HuM = cv2.HuMoments(m)
    for i in range(0,7):
        HuM[i] = -1*copysign(1.0, HuM[i]) * log10((abs(HuM[i])))
    return HuM

def get_Zernike_moments(cnt, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, color=255, thickness=cv2.FILLED)
    (x, y, w, h) = cv2.boundingRect(cnt)
    roi = mask[y:y + h, x:x + w]
    features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(cnt)[1], 8).tolist()
    return features

def get_instance_features(cnt, shape):
    instance_features = {}
    instance_features['circularity'] = get_circularity(cnt)
    instance_features['area'] = cv2.contourArea(cnt)
    instance_features['HuM'] = get_HU_moments(cnt).tolist()
    instance_features['Zernike'] = get_Zernike_moments(cnt, shape)
    ## ONDULACIÓN DEL PATCH
    ## OTROS DESCRIPTORES COMO:
    #       Fourier descriptor
    #       Curvature Scale Space
    #       SIFT
    #       Local phase quantization
    #       Multi-scale descriptors
    #       ?moments
    #       Chain code
    #       Distancia al centro, opuesta, apuntes cuaderno
    ##       Zernike
    #       MSER
    #       HOG
    return instance_features

def get_patch_features(contours, shape):
    patch_features = {}
    patch_features['n_instances'] = len(contours)
    patch_features['instances'] = {}
    for idx, cnt in enumerate(contours):
        patch_features['instances'][idx] = get_instance_features(cnt, shape)
    areas = [instance['area'] for instance in patch_features['instances'].values()]
    circularities = [instance['circularity'] for instance in patch_features['instances'].values()]
    patch_features['mean_area'] = np.mean(areas)
    patch_features['std_area'] = np.std(areas)
    patch_features['median_area'] = np.median(areas)
    patch_features['mean_circularity'] = np.mean(circularities)
    patch_features['std_circularity'] = np.std(circularities)
    patch_features['median_circularity'] = np.median(circularities)
    patch_features['outliers'] = {}
    patch_features['outliers']['area'] = []
    patch_features['outliers']['circularity'] = []
    # Outliers
    for idx, instance in patch_features['instances'].items():
        # Outliers will be computed with z scores
        # 2 is for 95 % of the data
        if abs((instance['area'] - patch_features['mean_area']) / patch_features['std_area']) > 2:
            patch_features['outliers']['area'].append({idx : instance['area']})
        if abs((instance['circularity'] - patch_features['mean_circularity']) / patch_features['std_circularity']) > 2:
            patch_features['outliers']['circularity'].append({idx : instance['circularity']})
    return patch_features

# EXAMPLE ON AVG_COLOUR
def main():
    img = cv2.imread('dataset_de_pacotilla\\1207.png')
    #    R = 255*np.ones((400,400), dtype=np.uint8)
    #    G = 169*np.ones((400,400), dtype=np.uint8)
    #    B = 255*np.ones((400,400), dtype=np.uint8)
    #    img = cv2.merge((B,G,R))
    cv2.imshow('Ey', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    colour = get_mean_contour_area(img)

    print(colour)

if __name__ == '__main__':
    main()