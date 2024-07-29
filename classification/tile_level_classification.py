from load_samples import TileSampleDataset
from sklearn.cluster import KMeans

DATA_PATH = "/home/gdem/Documents/Data/Processed_images_v2"


def main():
    dataset = TileSampleDataset(DATA_PATH)
    data = dataset.get_data()
    classifier = KMeans(2)
    classifier.fit(data[0])
    print(classifier.predict([dataset.get_patient('DFS', 25).X]))


if __name__ == '__main__':
    main()