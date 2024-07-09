import os
import shutil
import pandas as pd
import cv2
import numpy as np

def read_csv(csv_file):
    """Read the CSV file containing image paths and labels."""
    return pd.read_csv(csv_file)

def copy_images_to_folders(df, base_dir):
    """Copy images to folders based on their labels."""
    for index in range(len(df)):
        img_path = df.iloc[index, 0]  # Assuming the first column is the image path
        label = df.iloc[index, 1]     # Assuming the second column is the label
        label_dir = os.path.join(base_dir, str(label))

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        shutil.copy(img_path, label_dir)

def augment_image(img):
    """Apply augmentation to the image."""
    aug_list = []

    # Flip the image horizontally
    aug_list.append(cv2.flip(img, 1))

    # Rotate the image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        aug_list.append(rotated)

    # Add Gaussian noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    aug_list.append(noisy)

    return aug_list

def balance_dataset(base_dir, target_count):
    """Ensure each label folder contains an equal number of images using augmentation."""
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        images = os.listdir(label_dir)
        num_images = len(images)

        if num_images < target_count:
            print(f"Augmenting images in {label_dir} to reach {target_count} images.")
            while num_images < target_count:
                for img_name in images:
                    img_path = os.path.join(label_dir, img_name)
                    img = cv2.imread(img_path)
                    augmented_images = augment_image(img)

                    for aug_img in augmented_images:
                        aug_img_name = f"aug_{num_images}.jpg"
                        aug_img_path = os.path.join(label_dir, aug_img_name)
                        if cv2.imwrite(aug_img_path, aug_img):
                            num_images += 1
                        else:
                            print(f"Failed to write image: {aug_img_path}")
                            break

                        if num_images >= target_count:
                            break
                    if num_images >= target_count:
                        break
        print(f"Completed augmenting images in {label_dir}.")

def generate_csv(base_dir, output_csv):
    """Generate a CSV file with image paths and their corresponding labels."""
    data = []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            data.append([img_path, label])

    df = pd.DataFrame(data, columns=[None, None])  # No header
    df.to_csv(output_csv, index=False, header=False)

def main():
    csv_file = 'fundus-rs10922109_train.txt'  # Path to your CSV file
    base_dir = '/data/fundus-rs10922109_train'  # Base directory to store labeled folders
    target_count = 5000  # Target number of images per label

    output_csv = 'fundus-rs10922109_train_augmented.txt'  # Output CSV file path

    df = read_csv(csv_file)
    copy_images_to_folders(df, base_dir)
    balance_dataset(base_dir, target_count)
    generate_csv(base_dir, output_csv)

if __name__ == "__main__":
    main()
