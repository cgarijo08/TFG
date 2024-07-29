import os
from typing import List
import json

def _valid_path(dir):
    if os.path.exists(dir):
        return dir
    else:
        raise NotADirectoryError(f"Path: {dir} doesn't exists.")

class TileSample():
    def __init__(self, n, mean_area, std_area, mean_circ, std_circ, A, B, label, patient, tile):
        self.n = n
        self.mean_area = mean_area
        self.std_area = std_area
        self.mean_circ = mean_circ
        self.std_circ = std_circ
        self.A = A
        self.B = B
        self.X = [n, mean_area, std_area, mean_circ, std_circ, A, B]
        self.label = label
        self.patient = patient
        self.tile = int(tile)

class TileSampleDataset():
    def __init__(self, dir:str | os.PathLike):
        self.data_path = _valid_path(dir)
        self.samples:List[TileSample] = []
        self.load_data()
    
    def load_data(self):
        for pat in os.listdir(self.data_path):
            if pat == '0_logs':
                continue
            pat_path = os.path.join(self.data_path, pat)
            for tile in os.listdir(pat_path):
                tile_path = os.path.join(pat_path, tile)
                if os.path.exists(tile_path + '/patch_features.json'):
                    with open(tile_path + '/patch_features.json', 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    self.samples.append(TileSample(float(data['n_instances']), 
                                    data['mean_area'], 
                                    data['std_area'], 
                                    data['mean_circularity'], 
                                    data['std_circularity'], 
                                    data['avg_colour']['A'], 
                                    data['avg_colour']['B'],
                                    1,
                                    pat,
                                    tile))
    
    def get_data(self):
        X_data = []
        labels = []
        for sample in self.samples:
            X_data.append(sample.X)
            labels.append(sample.label)
        return [X_data, labels]
    
    def get_sample_from_patient(self, patient, tile):
        for sample in self.samples:
            if sample.patient == patient and sample.tile == tile:
                return sample
    
    def sample_from_idx(self, idx):
        return self.samples[idx]




class ContourSample():
    def __init__(self, area, circ, HuM:list, Zernike:List[float], label, patient, tile):
        self.area = area
        self.circ = circ
        self.HuM = [h[0] for h in HuM]
        self.Zernike = Zernike
        self.X = [area, circ, *self.HuM, *Zernike]
        self.label = label
        self.patient = patient
        self.tile = tile


class ContourSampleDataset():
    def __init__(self, dir:str | os.PathLike):
        self.data_path = _valid_path(dir)
        self.samples:List[TileSample] = []
        self.load_data()
    
    def load_data(self):
        for pat in os.listdir(self.data_path):
            if pat == '0_logs':
                continue
            pat_path = os.path.join(self.data_path, pat)
            for tile in os.listdir(pat_path):
                tile_path = os.path.join(pat_path, tile)
                if os.path.exists(tile_path + '/patch_features.json'):
                    with open(tile_path + '/patch_features.json', 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    for instance in data["instances"].values():
                        self.samples.append(ContourSample(
                            area=instance["area"],
                            circ=instance["circularity"],
                            HuM=instance["HuM"],
                            Zernike=instance["Zernike"],
                            label=1,
                            patient=pat,
                            tile=tile
                        ))
    
    def get_data(self):
        X_data = []
        labels = []
        for sample in self.samples:
            X_data.append(sample.X)
            labels.append(sample.label)
        return [X_data, labels]