from load_samples import ContourSampleDataset
from sklearn.decomposition import PCA

DATA_PATH = "/home/gdem/Documents/Data/Processed_images_v2"


def main():
    dataset = ContourSampleDataset(DATA_PATH)
    data = dataset.get_data()
    pca = PCA(10)
    reducted_data = pca.fit(data[0])
    print(reducted_data.explained_variance_ratio_)
    


if __name__ == '__main__':
    main()