import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report, confusion_matrix
from thundersvm import SVC
from torch.utils.data import DataLoader

from dataset import CustomDataset

# init svm
soft_svm = SVC(kernel="linear", C=0.1)
hard_svm = SVC(kernel="linear", C=100)
gaussian_svm = SVC(kernel="rbf")
sigmoid_svm = SVC(kernel="sigmoid")

svm = {
    "soft_svm": soft_svm,
    "hard_svm": hard_svm,
    "gaussian_svm": gaussian_svm,
    "sigmoid_svm": sigmoid_svm,
}


def extract_features(data_loader):
    model_name = "efficientnet-b4"

    net = EfficientNet.from_pretrained(model_name)
    print("Network loaded from pretrain")

    net.to(torch.device("cuda"))

    X_svm = []
    y_svm = []

    net.eval()
    with torch.no_grad():
        for X, y in data_loader:
            bs = X.size(0)

            # obtain data
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

            y_pred = net.extract_features(X)
            y_pred = nn.AdaptiveAvgPool2d(1)(y_pred)
            y_pred = y_pred.view(bs, -1)

            X_svm.extend(y_pred.tolist())
            y_svm.extend(y.tolist())

            print(len(X_svm), end="\r")

    return np.array(X_svm), np.array(y_svm)


def train_svm(X_svm, y_svm):
    # train and save
    for filename, model in svm.items():
        print("Start fitting {}".format(filename))
        model.fit(X_svm, y_svm)
        model.save_to_file("svm/" + filename)
    print("Done")


def test_svm(X_svm, y_svm):
    for filename, model in svm.items():
        # load
        model.load_from_file("svm/" + filename)

        # test and print result
        print("\n" + filename)
        print("Accuracy: {}".format(model.score(X_svm, y_svm)))
        y_pred = svm[filename].predict(X_svm)
        print(confusion_matrix(y_svm, y_pred))
        print(classification_report(y_svm, y_pred))


if __name__ == "__main__":
    # load data
    data = DataLoader(
        CustomDataset(phase="test", shape=(256, 256)),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # extract features
    features_vector, label = extract_features(data)

    # print test result
    test_svm(features_vector, label)
