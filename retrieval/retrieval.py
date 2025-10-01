import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def generate_retrieval_grid(query_img, retrieved_imgs, gt_labels, pred_labels, topk):
    border_size = 5
    spacing = 5
    img_size = (128, 128)

    total_images = topk + 1
    canvas_width = total_images * (img_size[0] + spacing)
    canvas_height = img_size[1] + 2 * border_size
    row_image = Image.new("RGB", (canvas_width, canvas_height), "white")

    all_imgs = [query_img] + retrieved_imgs
    correctness = [True] + [gt_labels[0] == pred for pred in pred_labels]

    for i, (img, correct) in enumerate(zip(all_imgs, correctness)):
        img = img.resize(img_size)
        bordered = Image.new("RGB", (img_size[0], img_size[1] + 2 * border_size), "white")
        bordered.paste(img, (0, border_size))

        draw = ImageDraw.Draw(bordered)
        color = "blue" if correct else "red"
        draw.rectangle(
            [(0, 0), (img_size[0] - 1, img_size[1] + 2 * border_size - 1)],
            outline=color,
            width=border_size,
        )

        x_offset = i * (img_size[0] + spacing)
        row_image.paste(bordered, (x_offset, 0))

    return row_image


def evaluate_retrieval(features_dict, y, image_paths, class_names, output_dir, top_k=15, num_queries=None):
    os.makedirs(output_dir, exist_ok=True)

    images = [Image.open(p).convert("RGB") for p in image_paths]
    num_samples = len(images)

    if num_queries is None:
        label_to_index = {}
        for idx, label in enumerate(y):
            if label not in label_to_index:
                label_to_index[label] = idx
                if len(label_to_index) == len(set(y)):
                    break
        indices_to_query = list(label_to_index.values())
    else:
        indices_to_query = np.random.choice(num_samples, num_queries, replace=False)

    method_scores = {name: [] for name in features_dict.keys()}

    for q_idx in tqdm(indices_to_query, desc="Generating retrieval results (1 per class)"):
        row_imgs = []
        row_labels = []
        accuracies = []

        for name, feats in features_dict.items():
            query_feat = feats[q_idx].reshape(1, -1)
            sims = cosine_similarity(query_feat, feats)[0]
            sorted_indices = np.argsort(sims)[::-1]
            retrieved_indices = [i for i in sorted_indices if i != q_idx][:top_k]

            query_img = images[q_idx]
            retrieved_imgs = [images[i] for i in retrieved_indices]
            gt_label = y[q_idx]
            pred_labels = [y[i] for i in retrieved_indices]
            correct = [1 if p == gt_label else 0 for p in pred_labels]
            acc = int(100 * sum(correct) / top_k)

            method_scores[name].append(acc)

            row = generate_retrieval_grid(query_img, retrieved_imgs, [gt_label]*top_k, pred_labels, top_k)
            row_imgs.append(row)
            row_labels.append(name)
            accuracies.append(f"{acc}%")

        font = ImageFont.load_default()
        row_height = row_imgs[0].height
        row_width = row_imgs[0].width
        label_width = 120
        score_width = 60
        total_width = label_width + row_width + score_width
        total_height = row_height * len(row_imgs)

        final_image = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(final_image)

        for i, (row_img, label, acc) in enumerate(zip(row_imgs, row_labels, accuracies)):
            y_offset = i * row_height
            draw.text((10, y_offset + row_height // 2 - 10), label, fill="black", font=font)
            final_image.paste(row_img, (label_width, y_offset))
            draw.text((label_width + row_width + 10, y_offset + row_height // 2 - 10), acc, fill="black", font=font)

        save_path = os.path.join(output_dir, f"retrieval_query{q_idx}.jpg")
        final_image.save(save_path)

    # === Affichage des scores moyens
    print("Mean Retrieval Accuracy per Feature:")
    for name, scores in method_scores.items():
        mean_score = sum(scores) / len(scores)
        print(f"  {name.upper():<10}: {mean_score:.2f}%")