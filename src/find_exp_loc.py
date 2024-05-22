import cv2
import numpy as np
import os

def extract_explanations(big_image_path, small_image_path, patches_dir='patches', min_patch_size=256, csv_file_path='patches_paths.txt'):
    # Read the big and small images
    big_image = cv2.imread(big_image_path)
    small_image = cv2.imread(small_image_path)

    if big_image is None:
        print(f"Error: Unable to load big image from path: {big_image_path}")
        return
    if small_image is None:
        print(f"Error: Unable to load small image from path: {small_image_path}")
        return

    # Compute convex hull points of the small image
    gray_small = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

    # Threshold parameters
    threshold = 80

    # Canny edge detection
    canny_output = cv2.Canny(gray_small, threshold, threshold * 2)

    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    # Calculate the scaling factors for x and y coordinates
    x_scale_factor = big_image.shape[1] / small_image.shape[1]
    y_scale_factor = big_image.shape[0] / small_image.shape[0]

    # Create the patches directory if it doesn't exist
    os.makedirs(patches_dir, exist_ok=True)

    # Prepare to write the paths of patches and the original big image to a text file
    with open(csv_file_path, 'w') as file:
        # Iterate over the hull list to extract patches and write paths
        for idx, hull in enumerate(hull_list):
            # Scale the hull points
            scaled_hull = np.copy(hull)
            scaled_hull[:, 0, 0] = (scaled_hull[:, 0, 0] * x_scale_factor).astype(float)
            scaled_hull[:, 0, 1] = (scaled_hull[:, 0, 1] * y_scale_factor).astype(float)

            # Use convex hull points to find the location in the big image
            x_min, y_min = np.min(scaled_hull[:, 0, :], axis=0)
            x_max, y_max = np.max(scaled_hull[:, 0, :], axis=0)

            # Ensure coordinates are within the bounds of the big image
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, big_image.shape[1])
            y_max = min(y_max, big_image.shape[0])

            patch_width = x_max - x_min
            patch_height = y_max - y_min

            # Check if the hull size is enough to contain at least a patch of the specified minimum size
            if patch_width < min_patch_size or patch_height < min_patch_size:
                print(f"Warning: Hull {idx} is smaller than the threshold ({min_patch_size}x{min_patch_size}). Skipping.")
                continue

            # Determine the number of patches to extract (up to 4)
            num_patches_x = patch_width // min_patch_size
            num_patches_y = patch_height // min_patch_size

            patches_extracted = 0
            for i in range(num_patches_x):
                for j in range(num_patches_y):
                    if patches_extracted >= 4:
                        break
                    x_start = x_min + i * min_patch_size
                    y_start = y_min + j * min_patch_size
                    x_end = x_start + min_patch_size
                    y_end = y_start + min_patch_size

                    if x_end > big_image.shape[1] or y_end > big_image.shape[0]:
                        continue

                    # Extract the patch from the big image using the coordinates
                    patch = big_image[y_start:y_end, x_start:x_end]

                    if patch.size == 0:
                        print(f"Warning: Patch {patches_extracted} is empty. Skipping.")
                        continue

                    # Save the patch
                    patch_image_path = os.path.join(patches_dir, f'extracted_patch_{idx}_{patches_extracted}.jpg')
                    cv2.imwrite(patch_image_path, patch)

                    # Write the paths to the CSV file
                    file.write(f'{patch_image_path},{big_image_path}\n')

                    patches_extracted += 1

            if patches_extracted == 0:
                print(f"Warning: No valid patches extracted from hull {idx}.")

    print(f'Paths of generated patches and the original big image written to {csv_file_path}')

if __name__ == '__main__':
    folder = 'patches'
    for path, subdirs, files in os.walk(folder):
        explanation = None
        original = None
        for name in files:
            fname=(os.path.join(path, name))
            if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.JPEG'):
                # Example usage
                if "explanation-found" in fname:
                    explanation = fname
                elif "Real_image" in fname:
                    original = fname
        if explanation and original:
            extract_explanations(original, explanation)
