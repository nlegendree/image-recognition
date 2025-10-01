import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from pathlib import Path

from utils.data_loader import get_dataloaders
from utils.plot import plot_confusion_matrix
from features.bow import sample_descriptors, encode_dataset_bow
from features.spatial_pyramid import encode_spatial_pyramid
from features.vgg import get_vgg_model, extract_vgg_feature
from classifiers.svm import train_svm
from classifiers.random_forest import train_random_forest
from classifiers.fc import train_fc_model
from retrieval.retrieval import evaluate_retrieval

def main():

    data_dir = "data/caltech20"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 0. Data loader
    train_loader, test_loader, train_paths, test_paths, train_labels, test_labels, class_names = get_dataloaders(data_dir)
    print(f"Dataset loaded with {len(class_names)} classes.")

    image_paths = train_paths + test_paths
    labels = train_labels + test_labels

    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))

    # 1-(1). 1. Sample descriptors
    descriptors = sample_descriptors(image_paths, sample_size=10000)

    # 1-(1). 2. PCA
    pca = PCA(n_components=64)
    desc_pca = pca.fit_transform(descriptors)

    # 1-(1). 3. KMeans
    kmeans = KMeans(n_clusters=100, random_state=42, n_init=10)
    kmeans.fit(desc_pca)

    # 1-(1). 4. Encode images into BoW
    X_bow, y = encode_dataset_bow(image_paths, labels, pca, kmeans)
    np.save(os.path.join(output_dir, "X_bow.npy"), X_bow)
    np.save(os.path.join(output_dir, "y.npy"), y)


    train_indices = np.arange(len(train_paths))
    test_indices = np.arange(len(train_paths), len(image_paths))
    np.savez(os.path.join(output_dir, "split_indices.npz"), train=train_indices, test=test_indices)


    # 1-(2) Spatial Pyramid Encoding
    X_spm = []
    for path in tqdm(image_paths, desc="Encoding SPM"):
        img = cv2.imread(path)
        vec = encode_spatial_pyramid(img, pca, kmeans, levels=[0, 1, 2])
        X_spm.append(vec)

    X_spm = np.array(X_spm)
    np.save(os.path.join(output_dir, "X_spm.npy"), X_spm)

    # 1-(3&4). VGG features
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_vgg13 = get_vgg_model("vgg13").to(device)
    X_vgg13 = []
    for path in tqdm(image_paths, desc="VGG13"):
        img = cv2.imread(path)
        vec = extract_vgg_feature(img, model_vgg13, device=device, augment=True)
        X_vgg13.append(vec)

    X_vgg13 = np.array(X_vgg13)
    np.save(os.path.join(output_dir, "X_vgg13.npy"), X_vgg13)

    model_vgg19 = get_vgg_model("vgg19").to(device)
    X_vgg19 = []
    for path in tqdm(image_paths, desc="VGG19"):
        img = cv2.imread(path)
        vec = extract_vgg_feature(img, model_vgg19, device=device, augment=True)
        X_vgg19.append(vec)

    X_vgg19 = np.array(X_vgg19)
    np.save(os.path.join(output_dir, "X_vgg19.npy"), X_vgg19)

    print("All datasets saved in 'output/'\n")

    # 2- Classifiers

    feature_files = {
        "bow": "X_bow.npy",
        "bowsp": "X_spm.npy",
        "vgg13": "X_vgg13.npy",
        "vgg19": "X_vgg19.npy"
    }
    y = np.load(os.path.join(output_dir, "y.npy"))
    split = np.load(os.path.join(output_dir, "split_indices.npz"))
    train_idx, test_idx = split["train"], split["test"]

    with open(os.path.join(output_dir, "class_names.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # This part is commented because it is implemented later in the code (3-(2))

    # 2-(1). Train SVM classifiers
    #for name, file in feature_files.items():
    #    path = os.path.join(output_dir, file)
    #    X = np.load(path)
    #    X_train, X_test = X[train_idx], X[test_idx]
    #    y_train, y_test = y[train_idx], y[test_idx]

    #    model_svm, acc_svm, cm_svm, y_true, y_pred = train_svm(X_train, y_train, X_test, y_test)

    # 2-(2). Train Random Forest classifiers
    #for name, file in feature_files.items():
    #    path = os.path.join(output_dir, file)
    #    X = np.load(path)
    #    X_train, X_test = X[train_idx], X[test_idx]
    #    y_train, y_test = y[train_idx], y[test_idx]

    #    model_rf, acc_rf, cm_rf, y_true, y_pred = train_random_forest(X_train, y_train, X_test, y_test)

    # 2-(3). Train Fully Connected classifiers
    #for name, file in feature_files.items():
    #    path = os.path.join(output_dir, file)
    #    X = np.load(path)
    #    X_train, X_test = X[train_idx], X[test_idx]
    #    y_train, y_test = y[train_idx], y[test_idx]
    #    input_dim = X.shape[1]
    #    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    #    X_test = (X_test - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    #    model_fc, acc_fc, cm_fc, y_true, y_pred = train_fc_model(X_train, y_train, X_test, y_test, input_dim=input_dim, device="cuda" if torch.cuda.is_available() else "cpu")


    # 3-(1). Image Retrieval with Encoded Vectors
    features_dict = {name: np.load(os.path.join(output_dir, fname)) for name, fname in feature_files.items()}
    evaluate_retrieval(
        features_dict=features_dict,
        y=y,
        image_paths=image_paths,
        class_names=class_names,
        output_dir=os.path.join(output_dir, "retrieval"),
        top_k=15,
        num_queries=20
    )

    # 3-(2). Classification accuracy
    Path(os.path.join(output_dir, "confmats")).mkdir(parents=True, exist_ok=True)
    for feat_name, file in feature_files.items():
        print(f"\nFeature: {feat_name.upper()}")
        X = np.load(os.path.join(output_dir, file))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_svm, acc_svm, cm_svm, y_true, y_pred = train_svm(X_train, y_train, X_test, y_test)
        print(f"SVM Accuracy ({feat_name.upper()}): {round(acc_svm * 100, 2)}%")
        plot_confusion_matrix(
            cm_svm,
            class_names,
            title=f"{feat_name.upper()} + SVM (Accuracy: {round(acc_svm * 100, 2)}%)",
            save_path=os.path.join(output_dir, "confmats", f"confmat_{feat_name}_svm.png")
        )

        model_rf, acc_rf, cm_rf, y_true, y_pred = train_random_forest(X_train, y_train, X_test, y_test)
        print(f"RF Accuracy ({feat_name.upper()}): {round(acc_rf * 100, 2)}%")
        plot_confusion_matrix(
            cm_rf,
            class_names,
            title=f"{feat_name.upper()} + RF (Accuracy: {round(acc_rf * 100, 2)}%)",
            save_path=os.path.join(output_dir, "confmats", f"confmat_{feat_name}_rf.png")
        )

        input_dim = X.shape[1]
        X_train_norm = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
        X_test_norm = (X_test - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
        model_fc, acc_fc, cm_fc, y_true, y_pred = train_fc_model(
            X_train_norm, y_train, X_test_norm, y_test, input_dim=input_dim, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"FC Accuracy ({feat_name.upper()}): {round(acc_fc * 100, 2)}%")
        plot_confusion_matrix(
            cm_fc,
            class_names,
            title=f"{feat_name.upper()} + FC (Accuracy: {round(acc_fc * 100, 2)}%)",
            save_path=os.path.join(output_dir, "confmats", f"confmat_{feat_name}_fc.png")
        )

if __name__ == "__main__":
    main()
