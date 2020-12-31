import argparse
import random
from warnings import filterwarnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dataset import load_label_dict

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mnb", help=["knn", "mnb", "svc", "gbc", "dtc"])
    parser.add_argument("--saved_model", type=str, default="")

    opt = parser.parse_args()

    if opt.saved_model != "":
        model = joblib.load(opt.saved_model)

        print("Load label dict...")
        label2ix, ix2label = load_label_dict("./data/label_dict.txt")

        print("Load test dataset...")
        test_vectors = sparse.load_npz("./data/test_vectors.npz")
        test_labels = np.load("./data/test_labels.npy")

        print("Predict test data...")
        y_pred = model.predict(test_vectors)

        report = classification_report(
            test_labels, y_pred,
            labels=list(range(len(ix2label))),
            target_names=ix2label
        )

        print(report)
    else:
        # choose model
        if opt.model == "knn":
            model = KNeighborsClassifier(n_jobs=4)
        elif opt.model == "mnb":
            model = MultinomialNB()
        elif opt.model == "svc":
            model = SVC()
        elif opt.model == "gbc":
            model = GradientBoostingClassifier()
        elif opt.model == "dtc":
            model = DecisionTreeClassifier()
        else:
            raise ValueError("Wrong model arg!")

        print("Load label dict...")
        label2ix, ix2label = load_label_dict("./data/label_dict.txt")

        print("Load train dataset...")
        train_vectors = sparse.load_npz("./data/train_vectors.npz")
        train_labels = np.load("./data/train_labels.npy")

        print("Load test dataset...")
        test_vectors = sparse.load_npz("./data/test_vectors.npz")
        test_labels = np.load("./data/test_labels.npy")

        print("Fit model...")
        model.fit(train_vectors, train_labels)

        print("Save model...")
        joblib.dump(model, f"./model/{opt.model}.model")

        print("Read model...")
        model = joblib.load(f"./model/{opt.model}.model")

        print("Predict test data...")
        y_pred = model.predict(test_vectors)

        report = classification_report(
            test_labels, y_pred,
            labels=list(range(len(ix2label))),
            target_names=ix2label
        )

        print("Save classification report...")
        with open(f"./report/{opt.model}.report", "w", encoding="utf8") as f:
            f.write(report)
        print(report)
