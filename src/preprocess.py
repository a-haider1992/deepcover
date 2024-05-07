import cv2
import numpy as np
import os
import pdb

def create_patches(image_path, patch_size):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # pdb.set_trace()
    height, width, _ = image.shape

    patches = []
    stride = patch_size // 2
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            if np.mean(patch) > 80:  # Check if patch contains significant region
                patches.append(patch)

    return patches

image_dir = "../data/Fundus_complete"
patch_dir = "/data/Fundus/patches"
pdb.set_trace()
os.makedirs(patch_dir, exist_ok=True)
patch_size = 128
with open("fundus_patches.txt", "w") as f:
    for root, dirs, files in os.walk(image_dir):
        identifier = root.split("/")[-1]
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                patches = create_patches(image_path, patch_size)
                image_name = os.path.splitext(filename)[0]
                output_dir = os.path.join(patch_dir, identifier, image_name)
                os.makedirs(output_dir, exist_ok=True)
                for i, patch in enumerate(patches):
                    patch_path = os.path.join(output_dir, f"patch_{i}.jpg")
                    cv2.imwrite(patch_path, patch)
                    f.write(os.path.join(output_dir, f"patch_{i}.jpg") + "," + identifier +"\n")

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
# patch_size = 128

# patches = create_patches(image_path, patch_size)
# os.makedirs("patches", exist_ok=True)
# for i, patch in enumerate(patches):
#     patch_path = f"patches/patch_{i}.jpg"
#     cv2.imwrite(patch_path, patch)
