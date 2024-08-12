import os
import math
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import models
from torch.autograd import Variable
from torchvision import transforms
from modules.network import ConvConvNext, ConvResNet, FCLayerConvNext, FCLayerResNet

# Paths
image_dir = '../Data-For-Segmentation/images/'
mask_dir = '../Data-For-Segmentation/masks/'
model_name = 'resnet'  # convnext

if model_name.lower() == 'convnext':
    circle_model_path = './models/convnext_tiny-1076-0.030622-maskIoU-0.938355.pth'
else:
    circle_model_path = './models/resnet18-027-0.008222-maskIoU-0.967159.pth'


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load model
if model_name.lower() == 'convnext':
    circle_model = models.convnext_tiny()
    circle_model.avgpool = ConvConvNext(in_channels=768, out_n=6)
    circle_model.classifier = FCLayerConvNext(in_h=7, in_w=10, out_n=6)
else:
    circle_model = models.resnet18()
    circle_model.avgpool = ConvResNet(in_channels=512, out_n=6)
    circle_model.fc = FCLayerResNet(out_n=6)

try:
    # Load model weights from file
    circle_model.load_state_dict(torch.load(circle_model_path, map_location=device))
except AssertionError:
    print("Assertion error occurred while loading model weights.")

circle_model.eval()

# Process images
filenames = os.listdir(image_dir)

for filename in filenames:
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    resized_image = image.resize((320, 240), Image.Resampling.BILINEAR)

    # Load and preprocess mask
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask).astype(float)

    # Disable gradient computation
    with torch.no_grad():
        # Apply input transformation, unsqueeze to add batch dimension, and move to device
        inp_xyr_t = circle_model(Variable(transform(resized_image).unsqueeze(0).repeat(1, 3, 1, 1).to(device)))

    # Circle parameters
    w, h = image.size
    diag = math.sqrt(w ** 2 + h ** 2)
    inp_xyr = inp_xyr_t.tolist()[0]  # Convert tensor to list
    pupil_x = int(inp_xyr[0] * w)
    pupil_y = int(inp_xyr[1] * h)
    pupil_r = int(inp_xyr[2] * 0.5 * 0.8 * diag)  # Adjusting radius based on image diagonal
    iris_x = int(inp_xyr[3] * w)
    iris_y = int(inp_xyr[4] * h)
    iris_r = int(inp_xyr[5] * 0.5 * diag)

    # Prepare image for visualization
    imVis = np.stack((np.array(image),) * 3, axis=-1)
    imVis[:, :, 1] = np.clip(imVis[:, :, 1] + 0.4 * mask_array, 0, 255)
    imVis[:, :, 2] = np.clip(imVis[:, :, 2] + 0.4 * mask_array, 0, 255)
    imVis = cv2.circle(imVis, (pupil_x, pupil_y), pupil_r, (0, 0, 255), 2)
    imVis = cv2.circle(imVis, (iris_x, iris_y), iris_r, (255, 0, 0), 2)

    # Show image
    cv2.imshow('Visualization', imVis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
