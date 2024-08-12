import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from modules.network import NestedSharedAtrousResUNet

# Define paths and directories
image_dir = "./input_images/"
seg_model_path = "./models/nestedsharedatrousresunet-025-0.133979-maskIoU-0.891476.pth"
output_dir = "./results/"

# Get a list of image files
files = glob.glob(os.path.join(image_dir, "*"))

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load segmentation model
model = NestedSharedAtrousResUNet(num_classes=1, num_channels=1, width=64)
model.load_state_dict(torch.load(seg_model_path, map_location=device))
model.eval()

# Process each image
for file in files:
    # Extract filename
    filename = os.path.basename(file)

    # Read image and resize
    image = Image.open(file).convert("L")
    width, height = image.size
    resized_image = image.resize((320, 240), Image.Resampling.BILINEAR)

    # Apply transformations
    tensor_image = transform(resized_image).unsqueeze(0).to(device)

    # Perform segmentation
    with torch.no_grad():
        mask_logit_t = model(tensor_image)[0]
        mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)

    # Convert mask tensor to numpy array
    mask = mask_t.cpu().numpy()[0]

    # Resize mask back to original dimensions
    mask = Image.fromarray(mask.astype("uint8"))
    mask = mask.resize((width, height), Image.Resampling.NEAREST)

    # Save mask
    mask.save(f"{output_dir}{filename}")
    print(filename)

