import cv2
import numpy as np
import pandas as pd
import os
import pdb
import shutil

def create_patches(image_path, patch_size):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # pdb.set_trace()
    height, width, _ = image.shape

    patches = []
    stride = patch_size // 4
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            total_num_pixels = patch_size * patch_size
            if np.sum(patch < 50) > 0.10 * total_num_pixels:
                continue
            patches.append(patch)
            # if np.mean(patch) > 95:  # Check if patch contains significant region
            #     count = np.sum(patch < 50)  # Count number of pixels above threshold
            #     if count > 0.5 * patch_size * patch_size:  # Check if count exceeds threshold
            #         continue  # Discard patch
            #     patches.append(patch)
            # if np.mean(patch) > 95:  # Check if patch contains significant region
            #     patches.append(patch)

    return patches

def create_masked_patches(image_path, patch_size):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width, _ = image.shape

    patches = []
    stride = patch_size // 4
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            total_num_pixels = patch_size * patch_size
            
            # Check if the number of pixels with intensity less than 50 is less than 10% of the total pixels
            if np.sum(patch < 50) > 0.10 * total_num_pixels:
                continue
            
            # Calculate the size of each black colored square patch
            black_patch_size = int(patch_size * 0.2)
            
            # Generate two random locations for black patches
            for _ in range(5):
                # Calculate the starting position of the black colored square patch
                black_patch_start_y = np.random.randint(0, patch_size - black_patch_size + 1)
                black_patch_start_x = np.random.randint(0, patch_size - black_patch_size + 1)
                
                # Create a black colored square patch
                black_patch = np.zeros_like(patch)
                black_patch[black_patch_start_y:black_patch_start_y+black_patch_size, black_patch_start_x:black_patch_start_x+black_patch_size] = 255
                
                # Overlay the black patch on the original patch
                patch = cv2.bitwise_or(patch, black_patch)
            
            patches.append(patch)

    return patches

def create_fundus_files(image_dir, patch_dir, patch_size):
    files_written = 0
    total_patches = 0
    with open("fundus_patches.txt", "w") as f:
        for root, dirs, files in os.walk(image_dir):
            identifier = root.split("/")[-1]
            files_written = 0
            for filename in files:
                patch_count_per_image = 0
                masked_patch_count_per_image = 0
                patch_written = 0
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(root, filename)
                    patches = create_patches(image_path, patch_size)
                    mask_patches = create_masked_patches(image_path, patch_size)
                    total_patches += len(patches)
                    patch_count_per_image += len(patches)
                    masked_patch_count_per_image += len(mask_patches)
                    image_name = os.path.splitext(filename)[0]
                    output_dir = os.path.join(patch_dir, identifier, image_name)
                    os.makedirs(output_dir, exist_ok=True)
                    for i, (patch, mask_patch) in enumerate(zip(patches, mask_patches)):
                        patch_path = os.path.join(output_dir, f"patch_{i}.jpg")
                        mask_patch_path = os.path.join(output_dir, f"mask_patch_{i}.jpg")
                        if cv2.imwrite(patch_path, patch) and cv2.imwrite(mask_patch_path, mask_patch):
                            f.write(f"{patch_path},{mask_patch_path},{identifier}\n")
                            files_written += 1
                            patch_written += 1
                    # for i, patch in enumerate(patches):
                    #     patch_path = os.path.join(output_dir, f"patch_{i}.jpg")
                    #     if cv2.imwrite(patch_path, patch):
                    #         f.write(os.path.join(output_dir, f"patch_{i}.jpg") + "," + identifier +"\n")
                    #         files_written += 1
                    #         patch_written += 1
        #         assert patch_count_per_image == patch_written, "Number of patches written does not match number of patches per image"
        #     print(f"Processed {identifier} with writing {files_written} patches")
        print(f"Total patches written: {total_patches}")


def train_test_split():
    with open("annotations.txt", "r") as f:
        pdb.set_trace()
        lines = f.readlines()
        np.random.shuffle(lines)
        split = int(0.8 * len(lines))
        train_lines = lines[:split]
        test_lines = lines[split:]

    with open("fundus_annotate_train.txt", "w") as f:
        for line in train_lines:
            f.write(line)

    with open("fundus_annotate_test.txt", "w") as f:
        for line in test_lines:
            f.write(line)

def normalize_id(id):
    return id.replace('I', '1')

if __name__ == "__main__":
    # image_dir = "../data/Fundus_complete"
    # image_dir = "/data/B-scans"
    oct_file = pd.read_csv("annotations-oct-sorted.txt", index_col=False, header=None, sep=',')

    # Define a function to extract the sorting keys
    # def extract_sorting_keys(path):
    #     try:
    #         parts = path.split('/')[3].split("_")
    #         if len(parts) >= 5:
    #             # Extract the second part, remove 'N' and convert to integer
    #             second_value = int(parts[1][1:])
    #             third_value = int(parts[4].split(".")[0])
    #             return second_value, third_value
    #     except (IndexError, ValueError) as e:
    #         print(f"Error processing path {path}: {e}")
    #     return float('inf'), float('inf')  # Handle cases where the path does not split correctly or conversion fails

    # # Apply the function to the first column and create new columns for sorting
    # oct_file[['sort_key_1', 'sort_key_2']] = oct_file.iloc[:, 0].apply(lambda x: pd.Series(extract_sorting_keys(x)))

    # # Print to check if keys are extracted correctly
    # print("Before sorting:")
    # print(oct_file[['sort_key_1', 'sort_key_2']].head(10))

    # # Sort the DataFrame based on the new columns
    # oct_file_sorted = oct_file.sort_values(by=['sort_key_1', 'sort_key_2'])

    # # Drop the sorting key columns if you don't need them anymore
    # oct_file_sorted = oct_file_sorted.drop(columns=['sort_key_1', 'sort_key_2'])

    # # Print to check if sorting worked
    # print("After sorting:")
    # print(oct_file_sorted.head(10))

    # # write the sorted DataFrame to a new file
    # oct_file_sorted.to_csv("annotations-oct-sorted.txt", header=False, index=False)

    # pdb.set_trace()

    # Read each line of the file
    for index, row in oct_file.iterrows():
        # Extract the image path and label from the row
        image_path, label = row[0], row[1]
        if label == 0.0:
            if not os.path.exists("data/0"):
                os.makedirs("data/0")
            shutil.move(image_path, "data/0")
        elif label == 1.0:
            if not os.path.exists("data/1"):
                os.makedirs("data/1")
            shutil.move(image_path, "data/1")
        elif label == 2.0:
            if not os.path.exists("data/2"):
                os.makedirs("data/2")
            shutil.move(image_path, "data/2")
        else:
            print(f"Invalid label {label} for image {image_path}")
    
    # with open("annotations-oct.txt", "w") as file:
    #     df = pd.read_csv("SNP_calls_combined_2020-06-16.csv", index_col=False, sep=',')
    #     # Filter the DataFrame for rows where gene value is "rs3750846"
    #     filtered_df = df[df['SNP'] == 'rs3750846']
    #     pdb.set_trace()
    #     nicola_ids = np.array(filtered_df.iloc[:, 0])
    #     labels = np.array(np.ceil(filtered_df.iloc[:, -1]))
    #     genes = np.array(filtered_df.iloc[:, 1])
    #     pdb.set_trace()

    #     for root, dirs, files in os.walk(image_dir):
    #         for filename in files:
    #             if filename.endswith(".jpg") or filename.endswith(".png"):
    #                 nicola_id = filename.split("_")[1]
    #                 normalized_nicola_id = normalize_id(nicola_id)
    #                 for id in nicola_ids:
    #                     normalized_id = normalize_id(id)
    #                     # pdb.set_trace()
    #                     if normalized_id == normalized_nicola_id:
    #                         print(f"Found {filename} with id {id} and gene {genes[nicola_ids == id][0]}")
    #                         file.write(os.path.join(root, filename) + "," + str(labels[nicola_ids == id][0]) + "\n")
    # patch_dir = "/data/Fundus/patches"
    # pdb.set_trace()
    # if os.path.exists(patch_dir):
    #     shutil.rmtree(patch_dir)
    # os.makedirs(patch_dir, exist_ok=True)
    # patch_size = 256
    # create_fundus_files(image_dir, patch_dir, patch_size)
    # os.makedirs(patch_dir)
    # patch_size = 256
    # train_test_split()
    # create_fundus_files(image_dir, patch_dir, patch_size)


    # for root, dirs, files in os.walk(image_dir):
    #     pdb.set_trace()
    #     for filename in files:
    #         if filename.endswith(".jpg") or filename.endswith(".png"):
    #             identifier = root.split("/")[-1]
    #             image_path = os.path.join(root, filename)
    #             patches = create_patches(image_path, patch_size)
    #             image_name = os.path.splitext(filename)[0]
    #             output_dir = os.path.join(patch_dir, identifier, image_name)
    #             os.makedirs(output_dir, exist_ok=True)
    #             for i, patch in enumerate(patches):
    #                 patch_path = os.path.join(output_dir, f"patch_{i}.jpg")
    #                 cv2.imwrite(patch_path, patch)
                # original_image_path = os.path.join(output_dir, f"{image_name}_original.jpg")
                # cv2.imwrite(original_image_path, cv2.imread(image_path))

    # image_path = "image7.jpg"
    # patch_size = 256

    # patches = create_masked_patches(image_path, patch_size)
    # os.makedirs("patches", exist_ok=True)
    # for i, patch in enumerate(patches):
    #     patch_path = f"patches/patch_{i}.jpg"
    #     cv2.imwrite(patch_path, patch)
