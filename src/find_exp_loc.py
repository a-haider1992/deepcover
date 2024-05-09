import cv2
import numpy as np
import pdb
# Read the big and small images
big_image = cv2.imread('image7.jpg')
small_image = cv2.imread('explanation-found-image7.jpg')

# pdb.set_trace()

# Convert the small image to grayscale
small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

# Use ORB feature detector and descriptor
orb = cv2.ORB_create()

# Find keypoints and descriptors in both images
kp1, des1 = orb.detectAndCompute(small_gray, None)
kp2, des2 = orb.detectAndCompute(big_image, None)

# Match keypoints between the two images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract corresponding keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate homography
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Transform coordinates of region in small image to big image
h, w = small_image.shape[:2]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, H)

# Plot dst as image
cv2.polylines(big_image, [np.int32(dst)], True, (0, 255, 0), 3)
cv2.imwrite('big_image.jpg', big_image)
# cv2.imshow('Region in Big Image', big_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Determine patch size
patch_size = (int(abs(dst[2][0][0] - dst[0][0][0])), int(abs(dst[2][0][1] - dst[0][0][1])))

print("Location of region in big image:", dst)
print("Patch size required:", patch_size)
