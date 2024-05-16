import cv2
import numpy as np
import os
import pdb
import shutil

def create_patches(image_path, patch_size):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # pdb.set_trace()
    height, width, _ = image.shape

    patches = []
    stride = patch_size // 8
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
    stride = patch_size // 2
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
        # print(f"Total patches written: {total_patches}")


def train_test_split():
    with open("fundus_patches.txt", "r") as f:
        pdb.set_trace()
        lines = f.readlines()
        np.random.shuffle(lines)
        split = int(0.8 * len(lines))
        train_lines = lines[:split]
        test_lines = lines[split:]

    with open("fundus_train.txt", "w") as f:
        for line in train_lines:
            f.write(line)

    with open("fundus_test.txt", "w") as f:
        for line in test_lines:
            f.write(line)


if __name__ == "__main__":
    # image_dir = "../data/Fundus_complete"
    # patch_dir = "/data/Fundus/masked_patches"
    # pdb.set_trace()
    # # if os.path.exists(patch_dir):
    # #     shutil.rmtree(patch_dir)
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

    image_path = "image7.jpg"
    patch_size = 256

    patches = create_masked_patches(image_path, patch_size)
    os.makedirs("patches", exist_ok=True)
    for i, patch in enumerate(patches):
        patch_path = f"patches/patch_{i}.jpg"
        cv2.imwrite(patch_path, patch)
