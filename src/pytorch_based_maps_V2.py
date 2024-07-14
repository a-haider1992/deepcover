import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import logging
from matplotlib import cm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

def get_img_array(img_path, size):
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.Resize([256, 256]),
      transforms.CenterCrop(256),  # Apply center crop
      CLAHETransform(clip_limit=1.0, tile_grid_size=(8, 8)),  # Apply CLAHE
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

    gradients = img_tensor.grad

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = features[0].detach()
    for i in range(pooled_gradients.size(0)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()

    return heatmap

def make_gradcampp_heatmap(model, img_tensor, class_idx=None):
    model.eval()  # Set model to evaluation mode

    # Lists to store activations and gradients
    features = []
    gradients = []

    # Forward hook to capture activations
    def forward_hook(module, input, output):
        features.append(output)
    
    # Backward hook to capture gradients
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # Find the last convolutional layer
    last_conv_layer = find_last_conv_layer(model)
    
    # Register hooks
    handle_forward = last_conv_layer.register_forward_hook(forward_hook)
    handle_backward = last_conv_layer.register_backward_hook(backward_hook)

    # Forward pass
    outputs = model(img_tensor)
    
    # Ensure hooks are removed before backward pass
    handle_forward.remove()
    handle_backward.remove()

    # If class_idx is not provided, use the index of the highest score
    if class_idx is None:
        class_idx = outputs.argmax(dim=1).item()
    
    # Compute the score for the target class
    class_score = outputs[0, class_idx]
    
    # Zero gradients and perform backward pass
    model.zero_grad()
    class_score.backward()
    
    # Extract gradients and activations
    if not gradients:
        raise RuntimeError("No gradients captured during backward pass.")
    gradients = gradients[0].detach()
    activations = features[0].detach()
    
    # ReLU on gradients
    with torch.no_grad():
        gradients = F.relu(gradients)
    
    # Ensure the shapes match
    if gradients.shape[1] != activations.shape[1]:
        raise ValueError("Shape mismatch: gradients and activations must have the same number of channels.")
    
    # Compute Grad-CAM++ weights
    alpha_num = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    weights = (alphas * F.relu(class_score.exp().detach())).sum(dim=(2, 3), keepdim=True)
    
    # Compute the heatmap
    heatmap = (weights * activations).sum(dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()
    
    return heatmap

def make_scorecam_heatmap(model, img_tensor, class_idx=None):
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

    activations = features[0].detach()
    activations = activations.cpu().numpy()

    # Get the score for the class
    with torch.no_grad():
        scores = []
        for i in range(activations.shape[1]):
            saliency_map = np.maximum(activations[0, i, :, :], 0)
            saliency_map = cv2.resize(saliency_map, (img_tensor.size(2), img_tensor.size(3)))
            saliency_map = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0)
            if img_tensor.is_cuda:
                saliency_map = saliency_map.cuda()
            score = model(img_tensor * saliency_map)
            scores.append(score[0, class_idx].item())

    scores = np.array(scores)
    scores = np.maximum(scores, 0)
    scores = scores / scores.sum() if scores.sum() != 0 else scores

    heatmap = (activations[0] * scores[:, np.newaxis, np.newaxis]).sum(axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def make_xgradcam_heatmap(model, img_tensor, class_idx=None):
    # Implementation for XGradCAM
    pass  # Replace with actual implementation if needed

def compute_gradcam(img_path, heatmap, activation_thresh, image_size, alpha=0.4):
    # Load the original image using PIL
    img_pil = Image.open(img_path)
    img_pil = img_pil.resize((image_size, image_size))
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = np.uint8(jet_heatmap * 255)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility

    # Resize heatmap to match the original image size
    jet_heatmap_resized = cv2.resize(jet_heatmap, (image_size, image_size))

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(jet_heatmap_resized, alpha, img, 1 - alpha, 0)

    # Convert back to PIL image for saving
    superimposed_img_pil = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    # Convert original image to grayscale and threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, int(activation_thresh * 255), 255, cv2.THRESH_BINARY)

    # Create mask for activation areas
    activation_mask = np.where(heatmap >= int(activation_thresh * 255), 1, 0).astype(np.uint8)
    activation_mask_resized = cv2.resize(activation_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    # Apply the activation mask to the original image
    img_activation = cv2.bitwise_and(img, img, mask=activation_mask_resized)

    # Combine activation areas with the superimposed image
    combined_img = cv2.addWeighted(img_activation, alpha, superimposed_img, 1 - alpha, 0)

    # Convert combined image back to PIL for saving
    combined_img_pil = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))

    return superimposed_img_pil, combined_img_pil

def process_images(images, output_dir, eobj, activation_thresh=0.2, image_size=256, alpha=0.4):
    model = eobj.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    import pdb
    for img_name in tqdm(images):
        # img_path = os.path.join(input_dir, img_name)
        img_tensor = get_img_array(img_name, image_size).to(device)

        for method in ['gradcam', 'scorecam']:
            if method == 'gradcam':
                heatmap = make_gradcam_heatmap(model, img_tensor)
            elif method == 'gradcampp':
                heatmap = make_gradcampp_heatmap(model, img_tensor)
            elif method == 'scorecam':
                heatmap = make_scorecam_heatmap(model, img_tensor)

            superimposed_img, combined_img = compute_gradcam(img_name, heatmap, activation_thresh, image_size, alpha)
            class_name = img_name.split("/")[2]
            image = img_name.split("/")[3].split(".")[0]
            superimposed_img_path = os.path.join(output_dir, class_name)
            real_path = os.path.join(output_dir, class_name, f"Real_image_{image}.png")
            combined_img_path = os.path.join(output_dir, class_name)
            if not os.path.exists(superimposed_img_path):
                os.makedirs(superimposed_img_path)
            superimposed_img_path = os.path.join(superimposed_img_path, f"{method}_{image}_superimposed.png")
            # if not os.path.exists(combined_img_path):
            #     os.mkdir(combined_img_path)
            #     combined_img_path = os.path.join(combined_img_path, "combined.png")

            superimposed_img.save(superimposed_img_path)
            real_image = cv2.imread(img_name)
            cv2.imwrite(real_path, real_image)
            # combined_img.save(combined_img_path)

