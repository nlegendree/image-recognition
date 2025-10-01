import cv2
import numpy as np

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray

def extract_dense_sift_for_spm(img_bgr, step_size=8, patch_size=16):
    img_gray = preprocess_gray(img_bgr)
    sift = cv2.SIFT_create()

    keypoints = []
    h, w = img_gray.shape
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            keypoints.append(cv2.KeyPoint(x, y, patch_size))

    keypoints, descriptors = sift.compute(img_gray, keypoints)
    return keypoints, descriptors

def encode_spatial_pyramid(img_bgr, pca, kmeans, levels=[0, 1, 2]):
    keypoints, desc = extract_dense_sift_for_spm(img_bgr)
    if desc is None or len(desc) == 0:
        return np.zeros(kmeans.n_clusters * (sum(4**l for l in levels)))

    desc_pca = pca.transform(desc)
    visual_words = kmeans.predict(desc_pca)

    h_img, w_img, _ = img_bgr.shape
    histograms = []

    for level in levels:
        num_cells = 2 ** level
        cell_h = h_img / num_cells
        cell_w = w_img / num_cells

        for i in range(num_cells):
            for j in range(num_cells):
                hist = np.zeros(kmeans.n_clusters)
                for kp, vw in zip(keypoints, visual_words):
                    x, y = kp.pt
                    if (i * cell_w <= x < (i + 1) * cell_w) and (j * cell_h <= y < (j + 1) * cell_h):
                        hist[vw] += 1
                if hist.sum() > 0:
                    hist /= hist.sum()  # Normalize each cell
                histograms.append(hist)

    final_histograms = []

    idx = 0
    for level in levels:
        num_cells = 4 ** level
        weight = {0: 0.25, 1: 0.25, 2: 0.5}[level]  # Customizable if needed
        for _ in range(num_cells):
            final_histograms.append(histograms[idx] * weight)
            idx += 1

    final_vec = np.concatenate(final_histograms)

    return final_vec