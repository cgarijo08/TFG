from load_samples import TileSampleDataset
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import os

DATA_PATH = "/home/gdem/Documents/Data/Processed_images_v2"

def _main():
    dataset = TileSampleDataset(DATA_PATH)
    data = dataset.get_data()
    patients = os.listdir(DATA_PATH)
    patients.pop(patients.index("0_logs"))
    patients = np.array(patients)
    np.random.seed(0)
    np.random.shuffle(patients)

    test_ratio = 0.45
    train_patients = patients[0:int((1-test_ratio)*len(patients))]
    test_patients = patients[int((1-test_ratio)*len(patients)):]

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for pat in train_patients:
        X_pat, Y_pat = dataset.get_samples_from_patient(pat)
        X_train.extend(X_pat)
        Y_train.extend(Y_pat)

    for pat in test_patients:
        X_pat, Y_pat = dataset.get_samples_from_patient(pat)
        X_test.extend(X_pat)
        Y_test.extend(Y_pat)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_neighbors = np.arange(2, 10)
    clf = KNeighborsClassifier()
    grid_clf = GridSearchCV(clf, {"n_neighbors": n_neighbors}, cv = 10)
    grid_clf.fit(X_train, Y_train)
    print(grid_clf.best_estimator_.score(X_train, Y_train))
    #print(clf.predict([dataset.get_sample_from_patient('DFS', 25).X]))
    best_clf = grid_clf.best_estimator_
    print(f"Best params: {grid_clf.best_params_}")
    y_pred = best_clf.predict(X_test)
    correct_predictions = np.sum(y_pred == Y_test)
    print(f"Correct predictions: {correct_predictions}/{len(Y_test)}")
    
    correct_patient_predictions = 0
    total_patients = 0

    correct_patient_predictions_0 = 0
    total_patients_0 = 0

    correct_patient_predictions_1 = 0
    total_patients_1 = 0

    correct_patient_predictions_2 = 0
    total_patients_2 = 0

    correct_patient_predictions_3 = 0
    total_patients_3 = 0

    for patient in test_patients:
        try:
            X_pat, Y_pat = dataset.get_samples_from_patient(patient)
            X_pat = scaler.transform(X_pat)
            Y_pred = best_clf.predict(X_pat)

            final_prediction = np.bincount(Y_pred).argmax()
            print(f"For patient: {patient}, the label predicted is: {final_prediction} while the ground truth is: {Y_pat[0]}")
            if final_prediction == Y_pat[0]:
                correct_patient_predictions += 1
            total_patients+=1

            if Y_pat[0] == 0:
                total_patients_0+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_0+=1
            elif Y_pat[0] == 1:
                total_patients_1+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_1+=1
            elif Y_pat[0] == 2:
                total_patients_2+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_2+=1
            elif Y_pat[0] == 3:
                total_patients_3+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_3+=1  
                    
        except:
            print(f"Patient: {patient} coulndt be processed")
    
    print(f"Predicted correctly {correct_patient_predictions} of {total_patients}\n\
          Accuracy: {correct_patient_predictions/total_patients}")
    print(f"For class 0: {correct_patient_predictions_0} of {total_patients_0}")
    print(f"For class 1: {correct_patient_predictions_1} of {total_patients_1}")
    print(f"For class 2: {correct_patient_predictions_2} of {total_patients_2}")
    print(f"For class 3: {correct_patient_predictions_3} of {total_patients_3}")

def ___main():
    dataset = TileSampleDataset(DATA_PATH)

    patients = os.listdir(DATA_PATH)
    patients.pop(patients.index("0_logs"))
    patients = np.array(patients)
    np.random.seed(0)
    np.random.shuffle(patients)

    test_ratio = 0.45
    train_patients = patients[0:int((1-test_ratio)*len(patients))]
    test_patients = patients[int((1-test_ratio)*len(patients)):]

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for pat in train_patients:
        X_pat, Y_pat = dataset.get_samples_from_patient(pat)
        X_train.extend(X_pat)
        Y_train.extend(Y_pat)

    for pat in test_patients:
        X_pat, Y_pat = dataset.get_samples_from_patient(pat)
        X_test.extend(X_pat)
        Y_test.extend(Y_pat)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #classifier = KMeans(2)
    #classifier.fit(data[0])
    n_neighbors = np.arange(2, 10)
    clf = KNeighborsClassifier()
    grid_clf = GridSearchCV(clf, {"n_neighbors": n_neighbors}, cv = 10)
    grid_clf.fit(X_train, Y_train)
    print(grid_clf.best_estimator_.score(X_train, Y_train))
    print(grid_clf.best_params_)
    #print(clf.predict([dataset.get_sample_from_patient('DFS', 25).X]))
    best_clf = grid_clf.best_estimator_
    print(f"Best params: {grid_clf.best_params_}")
    y_pred = best_clf.predict(X_test)
    correct_predictions = np.sum(y_pred == Y_test)
    print(f"Correct predictions: {correct_predictions}/{len(Y_test)}")

    correct_patient_predictions = 0
    total_patients = 0

    correct_patient_predictions_0 = 0
    total_patients_0 = 0

    correct_patient_predictions_1 = 0
    total_patients_1 = 0

    correct_patient_predictions_2 = 0
    total_patients_2 = 0

    correct_patient_predictions_3 = 0
    total_patients_3 = 0

    for patient in test_patients:
        try:
            X_pat, Y_pat = dataset.get_samples_from_patient(patient)
            X_pat = scaler.transform(X_pat)
            Y_pred = best_clf.predict(X_pat)

            final_prediction = np.bincount(Y_pred).argmax()
            print(f"For patient: {patient}, the label predicted is: {final_prediction} while the ground truth is: {Y_pat[0]}")
            if final_prediction == Y_pat[0]:
                correct_patient_predictions += 1
            total_patients+=1

            if Y_pat[0] == 0:
                total_patients_0+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_0+=1
            elif Y_pat[0] == 1:
                total_patients_1+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_1+=1
            elif Y_pat[0] == 2:
                total_patients_2+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_2+=1
            elif Y_pat[0] == 3:
                total_patients_3+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_3+=1  
                    
        except:
            print(f"Patient: {patient} coulndt be processed")
    
    print(f"Predicted correctly {correct_patient_predictions} of {total_patients}\n\
          Accuracy: {correct_patient_predictions/total_patients}")
    print(f"For class 0: {correct_patient_predictions_0} of {total_patients_0}")
    print(f"For class 1: {correct_patient_predictions_1} of {total_patients_1}")
    print(f"For class 2: {correct_patient_predictions_2} of {total_patients_2}")
    print(f"For class 3: {correct_patient_predictions_3} of {total_patients_3}")


def main():
    dataset = TileSampleDataset(DATA_PATH)

    patients = os.listdir(DATA_PATH)
    patients.pop(patients.index("0_logs"))
    patients = np.array(patients)
    np.random.seed(0)
    np.random.shuffle(patients)

    test_ratio = 0.45
    train_patients = patients[0:int((1-test_ratio)*len(patients))]
    test_patients = patients[int((1-test_ratio)*len(patients)):]

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for pat in train_patients:
        X_pat, Y_pat = dataset.get_samples_from_patient(pat)
        X_train.extend(X_pat)
        Y_train.extend(Y_pat)

    for pat in test_patients:
        X_pat, Y_pat = dataset.get_samples_from_patient(pat)
        X_test.extend(X_pat)
        Y_test.extend(Y_pat)
    print(f"Nº features:{len(X_test[0])}")
    print(f"Nº datos test: {len(X_test)}")
    print(f"Nº datos train: {len(X_train)}")
    print(f"Porcentaje de partición: {test_ratio*100}%")
    return 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #classifier = KMeans(2)
    #classifier.fit(data[0])
    n_neighbors = np.arange(2, 10)
    clf = OneVsRestClassifier(SVC())

    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_range = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])/np.array(X_test).shape[1]

    grid_clf = GridSearchCV(clf, {"estimator__C": C_range, "estimator__gamma":gamma_range}, cv = 10)
    grid_clf.fit(X_train, Y_train)
    print(grid_clf.best_estimator_.score(X_train, Y_train))
    print(grid_clf.best_params_)
    #print(clf.predict([dataset.get_sample_from_patient('DFS', 25).X]))
    best_clf = grid_clf.best_estimator_
    print(f"Best params: {grid_clf.best_params_}")
    y_pred = best_clf.predict(X_test)
    correct_predictions = np.sum(y_pred == Y_test)
    print(f"Correct predictions: {correct_predictions}/{len(Y_test)}")

    correct_patient_predictions = 0
    total_patients = 0

    correct_patient_predictions_0 = 0
    total_patients_0 = 0

    correct_patient_predictions_1 = 0
    total_patients_1 = 0

    correct_patient_predictions_2 = 0
    total_patients_2 = 0

    correct_patient_predictions_3 = 0
    total_patients_3 = 0

    for patient in test_patients:
        try:
            X_pat, Y_pat = dataset.get_samples_from_patient(patient)
            X_pat = scaler.transform(X_pat)
            Y_pred = best_clf.predict(X_pat)

            final_prediction = np.bincount(Y_pred).argmax()
            print(f"For patient: {patient}, the label predicted is: {final_prediction} while the ground truth is: {Y_pat[0]}")
            if final_prediction == Y_pat[0]:
                correct_patient_predictions += 1
            total_patients+=1

            if Y_pat[0] == 0:
                total_patients_0+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_0+=1
            elif Y_pat[0] == 1:
                total_patients_1+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_1+=1
            elif Y_pat[0] == 2:
                total_patients_2+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_2+=1
            elif Y_pat[0] == 3:
                total_patients_3+=1
                if final_prediction == Y_pat[0]:
                    correct_patient_predictions_3+=1  
                    
        except:
            print(f"Patient: {patient} coulndt be processed")
    
    print(f"Predicted correctly {correct_patient_predictions} of {total_patients}\n\
          Accuracy: {correct_patient_predictions/total_patients}")
    print(f"For class 0: {correct_patient_predictions_0} of {total_patients_0}")
    print(f"For class 1: {correct_patient_predictions_1} of {total_patients_1}")
    print(f"For class 2: {correct_patient_predictions_2} of {total_patients_2}")
    print(f"For class 3: {correct_patient_predictions_3} of {total_patients_3}")
if __name__ == '__main__':
    main()