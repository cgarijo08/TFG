import pandas as pd
import json

def main():
    samples = pd.read_csv("Apuntes_muestras.csv")
    patients = samples["PATIENT"]
    labels = samples["Classification"]
    classification = {}
    for patient, label in zip(patients, labels):
        classification[patient] = label
    with open("labels.json", 'w', encoding='utf-8') as file:
        json.dump(classification, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()