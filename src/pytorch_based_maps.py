import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import cv2
class CLAHETransform:
    def __init__(self, clip_limit=5.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        # Convert Tensor to PIL image if it's a tensor
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        # Convert PIL image to numpy array
        img = np.array(img)
        if len(img.shape) == 2:  # If the image is grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply CLAHE to each channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Convert numpy array back to PIL image
        img = Image.fromarray(img)
        # Convert PIL image back to Tensor
        # img = self.to_tensor(img)
        return img
    

def get_model_device(model):
    """
    Get the device of a PyTorch model.
    Handles models wrapped in DataParallel.
    """
    if isinstance(model, torch.nn.DataParallel):
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device

def generate_heatmap_image(img_boundry1, img_boundry2):
    heatmap_image = np.zeros_like(img_boundry1, dtype=np.uint8)
    heatmap_image[img_boundry1 > 0] = 255
    heatmap_image[img_boundry2 > 0] = 150
    return heatmap_image

def get_img_array(img_path, size):
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.Resize([256, 256]),
      transforms.CenterCrop(256),  # Apply center crop
      CLAHETransform(clip_limit=1.0, tile_grid_size=(8, 8)),  # Apply CLAHE
      # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

def find_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.
    """
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("No convolutional layer found in the model")

def make_gradcam_heatmap(model, img_tensor, class_idx=None):
    model.eval()
    img_tensor.requires_grad = True

    features = []
    def forward_hook(module, input, output):
        features.append(output)
    
    last_conv_layer = find_last_conv_layer(model)
    handle = last_conv_layer.register_forward_hook(forward_hook)

    outputs = model(img_tensor)
    handle.remove()

    if class_idx is None:
        class_idx = outputs.argmax(dim=1).item()

    class_score = outputs[0, class_idx]
    model.zero_grad()
    class_score.backward(retain_graph=True)

    gradients = img_tensor.grad.data
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = features[0].detach()
    for i in range(pooled_gradients.size(0)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()

    return heatmap

def compute_gradcam(img_path, heatmap, activation_thresh, image_size, alpha=0.4):
    # Load the original image using PIL
    img_pil = Image.open(img_path)
    img_pil = img_pil.resize((image_size, image_size))
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility

    # Rescale heatmap to a range 0-255 and apply jet colormap
    heatmap = np.uint8(255 * heatmap)
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize jet_heatmap to match the dimensions of the original image
    jet_heatmap_resized = cv2.resize(jet_heatmap, (image_size, image_size))

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(jet_heatmap_resized, alpha, img, 1 - alpha, 0)

    # Convert back to PIL image for saving
    superimposed_img_pil = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    # Convert original image to grayscale and threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, heatmask = cv2.threshold(img_gray, activation_thresh, 255, cv2.THRESH_BINARY)
    img_fg = cv2.bitwise_and(img, img, mask=heatmask)

    return heatmask, img_fg, superimposed_img_pil


def get_lime_explanation(img_path, model, image_size):
    logger.info('Generating LIME explanation for %s', img_path)

    def generate_heatmap_image(img_boundry1, img_boundry2):
        heatmap_image = np.zeros_like(img_boundry1, dtype=np.uint8)
        heatmap_image[img_boundry1 > 0] = 255
        heatmap_image[img_boundry2 > 0] = 150
        return heatmap_image

    # Load the image and ensure it is in the correct format
    img = Image.open(img_path).convert('RGB')
    img = img.resize((image_size, image_size))
    img = np.array(img)

    preprocess = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.Resize([256, 256]),
      transforms.CenterCrop(256),  # Apply center crop
      CLAHETransform(clip_limit=1.0, tile_grid_size=(8, 8)),  # Apply CLAHE
      # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def batch_predict(images):
        logger.info('Predicting batch of images with model')
        model.eval()
        batch = torch.stack([preprocess(Image.fromarray(image)) for image in images], dim=0)
        device = get_model_device(model)
        batch = batch.to(device)

        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img, batch_predict, top_labels=3, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp / 2 + 0.5, mask)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    img_boundry2 = mark_boundaries(temp / 2 + 0.5, mask)

    heatmap_image = generate_heatmap_image(img_boundry1, img_boundry2)
    return heatmap_image


def RG_sal_maps(image_path, image_size, model, activation_thresh=75):
    device = get_model_device(model)
    img_tensor = get_img_array(image_path, size=image_size).to(device)
    
    heatmap = make_gradcam_heatmap(model, img_tensor)

    lime_map = get_lime_explanation(image_path, model, image_size=image_size)

    heatmask, bitwise_map, superimposed_map = compute_gradcam(image_path, heatmap, activation_thresh=activation_thresh, image_size=image_size)
    return bitwise_map, superimposed_map, lime_map

def compute_gradcam_maps_V2(eobj, operation="BitwiseAND"):
    model = eobj.model
    all_dirs = eobj.fnames
    logger.info('Generating GradCAM maps for all images')
    output_dir = eobj.outputs
    THIS_DIR = os.getcwd()
    THIS_DIR = os.path.join(THIS_DIR, output_dir)

    if not os.path.exists(THIS_DIR):
        os.mkdir(THIS_DIR)

    for file in tqdm(all_dirs, desc="Computing maps"):
        bitwise_map, superimposed_map, lime_map = RG_sal_maps(file, eobj.image_size, model)
        class_name = file.split("/")[2]
        file_path = os.path.join(THIS_DIR, class_name)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        file_name_bitwise = "Bitwise_" + class_name +"_"+ file.split("/")[-1]
        file_name = os.path.join(file_path, file_name_bitwise)

        file_name_super = "Superimposed_" + class_name +"_"+ file.split("/")[-1]
        file_name_super = os.path.join(file_path, file_name_super)

        file_name_lime = "Lime_" + class_name +"_"+ file.split("/")[-1]
        file_name_lime = os.path.join(file_path, file_name_lime)

        original_image = "Real_image" + class_name +"_"+ file.split("/")[-1]
        original_image = os.path.join(file_path, original_image)

        logger.info('Saving Bitwise activation map for %s', file_name)
        logger.info('Saving Superimposed activation map for %s', file_name_super)
        logger.info('Saving Lime explanation for %s', file_name_lime)

        try:
            cv2.imwrite(file_name, bitwise_map)
        except Exception as e:
            print(f'Exception {e} while writing file {file_name}!')
        
        try:
            superimposed_map.save(file_name_super)
        except Exception as e:
            print(f'Exception {e} while writing file {file_name_super}!')
        
        try:
            cv2.imwrite(file_name_lime, lime_map)
        except Exception as e:
            print(f'Exception {e} while writing file {file_name_lime}!')

        try:
            img = cv2.imread(file)
            if cv2.imwrite(original_image, img):
                print("Real image saved successfully")
        except Exception as e:
            print(f'Exception {e} while writing file {original_image}!')
        
        continue