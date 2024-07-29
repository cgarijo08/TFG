import json
import argparse
import os
import numpy as np

def process_patient_features(patient_dir):
    mean_area = []
    std_area = []
    mean_circularity = []
    std_circularity = []
    color = []
    for tile in os.listdir(patient_dir):
        files = os.listdir(os.path.join(patient_dir, tile))
        if len(files) == 0:
            continue
        if "patch_features.json" in files:
            json_path = os.path.join(patient_dir, tile, 'patch_features.json')
        else:
            print(f"NO JSON FOUND. EXPECTED: {os.path.join(patient_dir, tile, 'patch_features.json')}")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mean_area.append(float(data['mean_area']))
        std_area.append(float(data['std_area']))
        mean_circularity.append(float(data['mean_circularity']))
        std_circularity.append(float(data['std_circularity']))
        color.append([float(data['avg_colour']['hue']), float(data['avg_colour']['A']), float(data['avg_colour']['B'])])
    print(np.mean(mean_area))
    print(np.mean(std_area))
    print(np.mean(mean_circularity))
    print(np.mean(std_circularity))
    print(np.mean(color, axis = 0))
def process_features(features_dir):
    for patient in os.listdir(features_dir):
        if patient == '0_logs':
            continue
        process_patient_features(os.path.join(features_dir, patient))
        break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dir", type=str, help="Directory where the patients shape images and json are.")
    args = parser.parse_args()


    if args.dir:
        features_dir = args.dir
    else:
        features_dir = '/home/gdem/Documents/Data/Processed_images_v2'

    process_features(features_dir)

if __name__ == '__main__':
    main()