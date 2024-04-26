import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import pdb
import cv2

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def RPN(image):
    '''
    image: cv2 image
    return: object_patches, boxes, labels
    '''
    def extract_object_patches(image, boxes):
        patches = []
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            patch = image[x_min: x_max, y_min: y_max]
            patches.append(patch)
        return patches
    # Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Transform the image
    input_image = transform(image).unsqueeze(0)

    # Get predictions from the model
    with torch.no_grad():
        predictions = model(input_image)

    boxes = predictions[0]['boxes'].tolist()
    labels = predictions[0]['labels'].tolist()

    # pdb.set_trace()

    object_patches = extract_object_patches(image, boxes)
    # print(f"Found {len(object_patches)} objects in the image")

    # for i, patch in enumerate(object_patches):
    #     cv2.imwrite(f"object_patch_{i}.jpg", patch)

    return object_patches, boxes, labels