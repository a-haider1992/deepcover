import cv2
import numpy as np
import pandas as pd
import os
import pdb
import shutil
import matplotlib.pyplot as plt

def extract_patches_from_heatmap(heatmap_path, original_image_path, output_dir, patch_size=256, max_patches=100, file_obj=None):
    """
    Extract patches from the original image at the locations of hot spots in the heatmap.
    
    Args:
        heatmap_path (str): Path to the heatmap image.
        original_image_path (str): Path to the original image.
        output_dir (str): Directory to save the extracted patches.
        patch_size (int): Size of the patches to extract.
        max_patches (int): Maximum number of patches to extract.
    """
    def extract_patch(image, center, patch_size=32):
        y, x = center
        half_size = patch_size // 2
        patch = image[y-half_size:y+half_size, x-half_size:x+half_size]
        total_num_pixels = patch_size * patch_size
        if np.sum(patch < 50) > 0.10 * total_num_pixels:
            return None
        else:
            return patch
    
    # Load the heatmap and original image
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path)

    # Get the dimensions of the heatmap and original image
    heatmap_height, heatmap_width = heatmap.shape
    original_image_height, original_image_width = original_image.shape[:2]

    # Threshold the heatmap to identify hotspots
    _, thresh_heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # Find the coordinates of the hotspots in the heatmap
    hotspots_coords = np.column_stack(np.where(thresh_heatmap > 0))

    # Scale coordinates to the original image's dimensions
    scale_x = original_image_width / heatmap_width
    scale_y = original_image_height / heatmap_height
    scaled_coords = [(int(y * scale_y), int(x * scale_x)) for y, x in hotspots_coords]

    # Ensure coordinates are in the valid range for patch extraction
    valid_coords = [
        (y, x) for (y, x) in scaled_coords 
        if y - patch_size // 2 >= 0 and y + patch_size // 2 <= original_image_height and 
           x - patch_size // 2 >= 0 and x + patch_size // 2 <= original_image_width
    ]

    # Extract patches and save them, limiting to max_patches
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    patch_count = 0
    for i, coord in enumerate(valid_coords):
        if patch_count >= max_patches:
            break
        patch = extract_patch(original_image, coord, patch_size)
        patch_filename = os.path.join(output_dir, f'patch_{patch_count}.png')
        if patch is not None:
            cv2.imwrite(patch_filename, patch)
            file_obj.write(f"{patch_filename}\n")
            patch_count += 1
            print(f"Saved patch to {patch_filename}")


def extract_patches_from_lime_map(lime_map_path, original_image_path, output_dir, patch_size=256, max_patches=100, file_obj=None):
    """
    Extract patches from the original image at the locations of green dots in the LIME map.
    
    Args:
        lime_map_path (str): Path to the LIME map image.
        original_image_path (str): Path to the original image.
        output_dir (str): Directory to save the extracted patches.
        patch_size (int): Size of the patches to extract.
    """
    def extract_patch(image, center, patch_size=32):
        y, x = center
        half_size = patch_size // 2
        patch = image[y-half_size:y+half_size, x-half_size:x+half_size]
        total_num_pixels = patch_size * patch_size
        # if np.sum(patch < 50) > 0.10 * total_num_pixels:
        #     return None
        # else:
        return patch
    
    # Load the LIME map and original image
    lime_map = cv2.imread(lime_map_path)
    original_image = cv2.imread(original_image_path)

    # Get the dimensions of the LIME map and original image
    lime_map_height, lime_map_width = lime_map.shape[:2]
    original_image_height, original_image_width = original_image.shape[:2]

    # Convert the LIME map to HSV color space
    hsv_lime_map = cv2.cvtColor(lime_map, cv2.COLOR_BGR2HSV)

    # Define the range for the green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask for the green color
    mask = cv2.inRange(hsv_lime_map, lower_green, upper_green)

    # Find the coordinates of the green dots in the LIME map
    green_dots_coords = np.column_stack(np.where(mask > 0))

    # pdb.set_trace()

    # Scale coordinates to the original image's dimensions
    scale_x = original_image_width / lime_map_width
    scale_y = original_image_height / lime_map_height
    scaled_coords = [(int(y * scale_y), int(x * scale_x)) for y, x in green_dots_coords]

    # Ensure coordinates are in the valid range for patch extraction
    valid_coords = [
        (y, x) for (y, x) in scaled_coords 
        if y - patch_size // 2 >= 0 and y + patch_size // 2 <= original_image_height and 
           x - patch_size // 2 >= 0 and x + patch_size // 2 <= original_image_width
    ]

    # Extract patches and save them
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    patch_count = 0
    for i, coord in enumerate(valid_coords):
        if patch_count >= max_patches:
            break
        patch = extract_patch(original_image, coord, patch_size)
        patch_filename = os.path.join(output_dir, f'patch_{i}.png')
        if patch is not None:
            cv2.imwrite(patch_filename, patch)
            file_obj.write(f"{patch_filename}\n")
            patch_count += 1
            print(f"Saved patch to {patch_filename}")


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


def train_val_test_split():
    with open("annotations-oct-sorted.txt", "r") as f:
        pdb.set_trace()
        lines = f.readlines()
        # np.random.shuffle(lines)
        
        train_split = int(0.7 * len(lines))
        val_split = int(0.8 * len(lines))  # 70% + 10% = 80%

        train_lines = lines[:train_split]
        val_lines = lines[train_split:val_split]
        test_lines = lines[val_split:]

    with open("oct_annotate_train.txt", "w") as f:
        for line in train_lines:
            f.write(line)

    with open("oct_annotate_val.txt", "w") as f:
        for line in val_lines:
            f.write(line)

    with open("oct_annotate_test.txt", "w") as f:
        for line in test_lines:
            f.write(line)

def train_test_split(id_list):
    print(f"Received {len(id_list)} unique IDs")
    
    # Load the data from the CSV file
    data = pd.read_csv("fundus-rs570618.txt", header=None, index_col=False, sep=',')
    lines = data.values[:, 0]  # Assuming the first column contains the file paths
    labels = data.values[:, 1]  # Assuming the second column contains the labels

    # Create a dictionary to map IDs to (line, label) tuples
    id_to_lines = {}
    for idx, line in enumerate(lines):
        id = line.split("/")[-1].split("_")[0]  # Assuming ID is the first element in each line
        id = normalize_id(id)
        if id in id_to_lines:
            id_to_lines[id].append((line, labels[idx]))
        else:
            id_to_lines[id] = [(line, labels[idx])]

    print(f"Found {len(id_to_lines)} unique IDs")
    
    # Filter the dictionary to only include lines with IDs in id_list
    id_to_lines = {k: v for k, v in id_to_lines.items() if k in id_list}
    
    # Shuffle the IDs
    ids = list(id_to_lines.keys())
    np.random.shuffle(ids)
    
    # Split the IDs into train and test sets
    train_split = int(0.8 * len(ids))
    train_ids = ids[:train_split]
    test_ids = ids[train_split:]
    
    # Collect lines for train and test sets
    train_lines = [item for id in train_ids for item in id_to_lines[id]]
    test_lines = [item for id in test_ids for item in id_to_lines[id]]
    
    print(f"Train lines: {len(train_lines)}, Test lines: {len(test_lines)}")

    # Write the lines to the train and test files
    with open("fundus-rs570618_train.txt", "w") as f:
        for line, label in train_lines:
            f.write(f"{line},{label}\n")
    
    with open("fundus-rs570618_test.txt", "w") as f:
        for line, label in test_lines:
            f.write(f"{line},{label}\n")

def train_test_split_V2(id_list):
    print(f"Received {len(id_list)} unique IDs")
    
    # Load the data from the CSV file
    data = pd.read_csv("oct-rs570618.txt", header=None, index_col=False, sep=',')
    lines = data.values[:, 0]  # Assuming the first column contains the file paths
    labels = data.values[:, 1]  # Assuming the second column contains the labels

    # Create a dictionary to map IDs to (line, label) tuples
    id_to_lines = {}
    # pdb.set_trace()
    for idx, line in enumerate(lines):
        id = line.split("/")[-1].split("_")[1]  # Assuming ID is the first element in each line
        id = normalize_id(id)
        if id in id_to_lines:
            id_to_lines[id].append((line, labels[idx]))
        else:
            id_to_lines[id] = [(line, labels[idx])]

    print(f"Found {len(id_to_lines)} unique IDs")
    
    # Filter the dictionary to only include lines with IDs in id_list
    id_to_lines = {k: v for k, v in id_to_lines.items() if k in id_list}
    
    # Shuffle the IDs
    ids = list(id_to_lines.keys())
    np.random.shuffle(ids)
    
    # Split the IDs into train, validation, and test sets
    train_split = int(0.7 * len(ids))
    validation_split = int(0.1 * len(ids)) + train_split
    train_ids = ids[:train_split]
    validation_ids = ids[train_split:validation_split]
    test_ids = ids[validation_split:]
    
    # Collect lines for train, validation, and test sets
    train_lines = [item for id in train_ids for item in id_to_lines[id]]
    validation_lines = [item for id in validation_ids for item in id_to_lines[id]]
    test_lines = [item for id in test_ids for item in id_to_lines[id]]
    
    print(f"Train lines: {len(train_lines)}, Validation lines: {len(validation_lines)}, Test lines: {len(test_lines)}")

    # Write the lines to the train, validation, and test files
    with open("oct-rs570618_train.txt", "w") as f:
        for line, label in train_lines:
            f.write(f"{line},{label}\n")
    
    with open("oct-rs570618_validation.txt", "w") as f:
        for line, label in validation_lines:
            f.write(f"{line},{label}\n")

    with open("oct-rs570618_test.txt", "w") as f:
        for line, label in test_lines:
            f.write(f"{line},{label}\n")


def normalize_and_save_images(main_folder, output_main_folder, weights=[0.5, 0.5, 0.5]):
    def load_images(image_dir):
        images = []
        file_paths = []
        for root, dirs, files in os.walk(image_dir):
            for filename in files:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)  # Load in color
                    if img is not None:
                        images.append(img)
                        file_paths.append(img_path)
        return images, file_paths

    def calculate_mean_std(images):
        brightness_values = []
        for img in images:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness_values.append(hsv_img[:, :, 2].flatten())  # V channel for brightness
        brightness_values = np.concatenate(brightness_values)
        mean = np.mean(brightness_values)
        std = np.std(brightness_values)
        return mean, std

    def normalize_image(image, mean_src, std_src, mean_target, std_target):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv_img[:, :, 2]
        normalized_v = (v_channel - mean_src) / std_src * std_target + mean_target
        normalized_v = np.clip(normalized_v, 0, 255)  # Ensure pixel values are valid
        hsv_img[:, :, 2] = normalized_v.astype(np.uint8)
        normalized_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return normalized_img

    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    output_subfolders = [os.path.join(output_main_folder, os.path.basename(subfolder)) for subfolder in subfolders]

    # Calculate the global mean and standard deviation
    all_images = []
    for subfolder in subfolders:
        images, _ = load_images(subfolder)
        all_images.extend(images)
    global_mean, global_std = calculate_mean_std(all_images)

    for subfolder, output_subfolder, weight in zip(subfolders, output_subfolders, weights):
        images, paths = load_images(subfolder)
        
        mean, std = calculate_mean_std(images)
        
        # Adjust the target mean and std with weights
        adjusted_mean_target = global_mean * (weight) + mean * (1 - weight)
        adjusted_std_target = global_std * (weight) + std * (1 - weight)
        
        os.makedirs(output_subfolder, exist_ok=True)
        
        for idx, img in enumerate(images):
            normalized_img = normalize_image(img, mean, std, adjusted_mean_target, adjusted_std_target)
            output_path = os.path.join(output_subfolder, os.path.basename(paths[idx]))
            cv2.imwrite(output_path, normalized_img)


def normalize_id(id):
    return id.replace('I', '1')

if __name__ == "__main__":
    # image_dir = "../data/Fundus_complete"
    image_dir = "/data/B-scans"
    # oct_file = pd.read_csv("oct-rs570618_train.txt", index_col=False, header=None, sep=',')
    patch_dir = "/data/oct-timebased"
    # image_dir = "../outs-cam"

    # output_dir = '../data/Fundus_normalized'
    # normalize_and_save_images(image_dir, output_dir)

    # Assuming the second column contains the class labels
    # labels = oct_file[1]

    # # Count the occurrences of each class
    # class_counts = labels.value_counts().sort_index()
    # # Plot the histogram
    # plt.figure(figsize=(8, 6))
    # plt.bar(class_counts.index, class_counts.values, tick_label=['Class 0', 'Class 1', 'Class 2'])
    # plt.xlabel('Class')
    # plt.ylabel('Number of Images')
    # plt.title('Distribution of Images Across Classes')
    # plt.xticks(range(3))
    # plt.savefig("fundus_traindata_distribution.png")

    # output_dir = 'lime_extracted_patch'
    # gradcam_map_path = '../outs-cam/1/Superimposed_1_TE-O_17_20140415111657635_R.jpg'
    # original_image_path = '../data/Fundus_correct/1/TE-O_17_20140415111657635_R.jpg'
    # extract_patches_from_heatmap(gradcam_map_path, original_image_path, output_dir)
    # with open("lime_gradcam_patches.txt", "a+") as f:
    #     for root, dirs, files in os.walk(image_dir):
    #         for dir in dirs:
    #             print(f"Processing directory {dir}")
    #             # pdb.set_trace()
    #             files = os.listdir(os.path.join(root, dir))
    #             for filename in files:
    #                 # print(f"Processing {filename}")
    #                 if filename.startswith("Lime"):
    #                     print(f"Processing {filename}")
    #                     # Example usage
    #                     original_image_name = "TE-"+filename.split("-")[1]
    #                     lime_map_path = os.path.join(root, dir, filename)
    #                     original_image_path = os.path.join('../data/Fundus_correct', dir, original_image_name)
    #                     # pdb.set_trace()
    #                     # lime_map_path = '../outs-cam/0/Lime_0_TE-Z_1_20140131114839533_R.jpg'
    #                     # original_image_path = '../data/Fundus_correct/0/TE-Z_1_20140131114839533_R.jpg'
    #                     output_dir = 'lime_extracted_patch'
    #                     original_image_name_no_ext = original_image_name.split(".")[0]
    #                     if dir == '0':
    #                         output_dir = f'{output_dir}/0/{original_image_name_no_ext}'
    #                     elif dir == '1':
    #                         output_dir = f'{output_dir}/1/{original_image_name_no_ext}'
    #                     elif dir == '2':
    #                         output_dir = f'{output_dir}/2/{original_image_name_no_ext}'
    #                     else:
    #                         raise ValueError(f"Invalid directory {dir}")
    #                     extract_patches_from_lime_map(lime_map_path, original_image_path, output_dir, file_obj=f)
                        # extract_patches_from_heatmap(lime_map_path, original_image_path, output_dir, file_obj=f)

    # train_val_test_split()

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

    # Define the number of samples per class
    # n_samples_per_class = 136 // 3  # This assumes you want an equal number from each class

    # # Split the dataset based on the label
    # class_0 = oct_file[oct_file[1] == 0.0]
    # class_1 = oct_file[oct_file[1] == 1.0]
    # class_2 = oct_file[oct_file[1] == 2.0]

    # # Randomly sample the desired number of files from each class
    # sampled_class_0 = class_0.sample(n=n_samples_per_class, random_state=42)
    # sampled_class_1 = class_1.sample(n=n_samples_per_class, random_state=42)
    # sampled_class_2 = class_2.sample(n=n_samples_per_class, random_state=42)

    # # Combine the sampled data
    # sampled_oct_file = pd.concat([sampled_class_0, sampled_class_1, sampled_class_2])

    # for index, row in sampled_oct_file.iterrows():
    #     # Extract the image path and label from the row
    #     image_path, label = row[0], row[1]
    #     if label == 0.0:
    #         if not os.path.exists("../data/Fundus/0"):
    #             os.makedirs("../data/Fundus/0")
    #         shutil.copy(image_path, "../data/Fundus/0")
    #     elif label == 1.0:
    #         if not os.path.exists("../data/Fundus/1"):
    #             os.makedirs("../data/Fundus/1")
    #         shutil.copy(image_path, "../data/Fundus/1")
    #     elif label == 2.0:
    #         if not os.path.exists("../data/Fundus/2"):
    #             os.makedirs("../data/Fundus/2")
    #         shutil.copy(image_path, "../data/Fundus/2")
    #     else:
    #         print(f"Invalid label {label} for image {image_path}")

    # # Read each line of the file
    # for index, row in oct_file.iterrows():
    #     # Extract the image path and label from the row
    #     image_path, label = row[0], row[1]
    #     if label == 0.0:
    #         if not os.path.exists("/data/test/0"):
    #             os.makedirs("/data/test/0")
    #         shutil.move(image_path, "/data/test/0")
    #         print(f"Copying {image_path} to /data/test/0")
    #     elif label == 1.0:
    #         if not os.path.exists("/data/test/1"):
    #             os.makedirs("/data/test/1")
    #         shutil.move(image_path, "/data/test/1")
    #         print(f"Copying {image_path} to /data/test/1")
    #     elif label == 2.0:
    #         if not os.path.exists("/data/test/2"):
    #             os.makedirs("/data/test/2")
    #         shutil.move(image_path, "/data/test/2")
    #         print(f"Copying {image_path} to /data/test/2")
    #     else:
    #         print(f"Invalid label {label} for image {image_path}")
    
    # with open("oct-rs570618.txt", "w") as file:
    #     df = pd.read_csv("SNP_calls_combined_2021-03-04.csv", index_col=False, sep=',')
    #     # Filter the DataFrame for rows where gene value is "rs3750846"
    #     filtered_df = df[df['SNP'] == 'rs570618']
    #     print(f"Length of filtered df: {len(filtered_df)}")
    #     # print(f"Found {len(filtered_df)} rows with gene value 'rs10922109'")
    #     # pdb.set_trace()
    #     nicola_ids = np.array(filtered_df.iloc[:, 0])
    #     print(f"Found {len(np.unique(nicola_ids))} ids")
    #     # Ceil the labels to convert them to integers
    #     labels = np.array(np.ceil(filtered_df.iloc[:, -1]))
    #     genes = np.array(filtered_df.iloc[:, 1])
    #     # pdb.set_trace()

    #     files_written = []

    #     for root, dirs, files in os.walk(image_dir):
    #         for filename in files:
    #             if filename.endswith(".jpg") or filename.endswith(".png"):
    #                 nicola_id = filename.split("_")[1]
    #                 normalized_nicola_id = normalize_id(nicola_id)
    #                 for id in nicola_ids:
    #                     normalized_id = normalize_id(id)
    #                     # pdb.set_trace()
    #                     if normalized_id == normalized_nicola_id:
    #                         # print(f"Found {filename} with id {id} and gene {genes[nicola_ids == id][0]}")
    #                         file.write(os.path.join(root, filename) + "," + str(labels[nicola_ids == id][0]) + "\n")
    #                         files_written.append(filename)
    #     print(f"Files written : {len(files_written)}")

    # patch_dir = "/data/Fundus/patches"
    # pdb.set_trace()
    # if os.path.exists(patch_dir):
    #     shutil.rmtree(patch_dir)
    # os.makedirs(patch_dir, exist_ok=True)
    # patch_size = 256
    # create_fundus_files(image_dir, patch_dir, patch_size)
    # os.makedirs(patch_dir)
    # patch_size = 256
    # df = pd.read_csv("SNP_calls_combined_2021-03-04.csv", index_col=False, sep=',')
    # Filter the DataFrame for rows where gene value is "rs3750846"
    # filtered_df = df[df['SNP'] == 'rs570618']
    # print(f"Length of filtered df: {len(filtered_df)}")
    # nicola_ids = np.array(filtered_df.iloc[:, 0])
    # print(f"Found {len(np.unique(nicola_ids))} ids")
    # train_test_split_V2(id_list=nicola_ids)
    # create_fundus_files(image_dir, patch_dir, patch_size)

    patients = []
    files_copied = 0
    for root, dirs, files in os.walk(image_dir):
        pdb.set_trace()
        df = pd.read_csv("/data/Layer-thickness/All_ThicknessMap_Retina.csv", index_col=False, sep=',')
        print(f"Number of files in directory: {len(files)}")
        assert len(files) > 0, "No files found in the directory"
        assert len(files) == 641749, "Number of files are less than 641749"
        for filename in files:
            output_dir = None
            if filename.endswith(".jpg") or filename.endswith(".png"):
                patient_id = filename.split("_")[1]
                # patient_id = normalize_id(patient_id)
                pid = filename.split("_")[2]

                # if patient_id == "N10165":
                #     pdb.set_trace()
                
                # if patient_id not in patients:
                patients.append(patient_id)
                filtered_df = df[df['Firstname'] == patient_id]
                
                if not filtered_df.empty:
                    pid2 = filtered_df['ImageID'].values.astype(str)
                    
                    for i, p in enumerate(pid2):
                        eye = filtered_df.iloc[i]['Eye']
                        output_dir = os.path.join(patch_dir, patient_id, eye)
                        print(f"Found patient {patient_id} with eye {eye}")
                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        if p == pid:
                            src_path = os.path.join(root, filename)
                            dst_path = os.path.join(output_dir, filename)
                            shutil.move(src_path, dst_path)
                            print(f"Moved {filename} to {output_dir}")
                            files_copied += 1

        print(f"Patients : {len(np.unique(patients))}")
        print(f"Files copied: {files_copied}")
        assert files_copied == len(files), "Number of files copied does not match number of files in the directory"

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
