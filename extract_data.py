import os, cv2
from shutil import copy2, move
import numpy as np
import zipfile

def extract_dataset():
    print("[INFO] Extracting data...")

    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall()

    print(f"[INFO] Data extracted at ./dataset")

def get_dataset_statistics(DATASET_PATH = "./dataset"):

    print(f"[INFO] Calculating data statistics...")

    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir = os.path.join(DATASET_PATH, "val")

    for i in [train_dir, val_dir]:
        min_h = np.inf
        max_h = 0
        avg_h = 0

        min_w = np.inf
        max_w = 0
        avg_w = 0

        count = 0

        for class_name in os.listdir(i):
            class_path = os.path.join(i, class_name)
            class_images = os.listdir(class_path)

            for image_path in class_images:
                h, w, c = cv2.imread(os.path.join(class_path, image_path)).shape
                avg_h += h
                avg_w += w

                min_h = min(min_h, h)
                min_w = min(min_w, w)
                max_h = max(max_h, h)
                max_w = max(max_w, w)
                count += 1

        avg_h /= count
        avg_w /= count

        if i == train_dir:
            print("Training Dataset Statistics:")
        else:
            print("Validation Dataset Statistics:")

        print(f'Height Range: {min_h} --> {max_h}\tAverage: {"%.2f" %avg_h}')
        print(f'Width Range: {min_w} --> {max_w}\tAverage: {"%.2f" %avg_w}\n')
