import cv2
import numpy as np
from tqdm import tqdm

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray

def extract_dense_sift(img_bgr, step_size=8, patch_size=16):
    img_gray = preprocess_gray(img_bgr)
    sift = cv2.SIFT_create()
    keypoints = []
    h, w = img_gray.shape
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            keypoints.append(cv2.KeyPoint(x, y, patch_size))
    keypoints, descriptors = sift.compute(img_gray, keypoints)
    return descriptors

def sample_descriptors(image_paths, sample_size=10000, step_size=8):
    all_desc = []
    for path in tqdm(image_paths, desc="Extracting SIFT"):
        img = cv2.imread(path)
        desc = extract_dense_sift(img, step_size=step_size)
        if desc is not None:
            all_desc.append(desc)
    all_desc = np.vstack(all_desc)
    np.random.shuffle(all_desc)
    return all_desc[:sample_size]

def encode_image_bow(img_bgr, pca, kmeans, step_size=8):
    desc = extract_dense_sift(img_bgr, step_size=step_size)
    if desc is None or len(desc) == 0:
        return np.zeros(kmeans.n_clusters)
    desc_pca = pca.transform(desc)
    visual_words = kmeans.predict(desc_pca)
    hist, _ = np.histogram(visual_words, bins=np.arange(kmeans.n_clusters + 1), density=True)
    return hist

def encode_dataset_bow(image_paths, labels, pca, kmeans, step_size=8):
    X = []
    y = []
    for path, labels in tqdm(zip(image_paths, labels), desc="Encoding BoW", total=len(image_paths)):
        img = cv2.imread(path)
        vec = encode_image_bow(img, pca, kmeans, step_size=step_size)
        X.append(vec)
        y.append(labels)
    return np.array(X), np.array(y)