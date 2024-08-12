import os
import glob
from PIL import Image

input_dir = '/Users/raselahmedbhuiyan/Documents/cropping-iris-images-manually/Newborn-Iris-Dataset/'
output_dir = '/Users/raselahmedbhuiyan/Documents/New-born-Iris-Segmentation/Piotr-Newborn-Iris-Cropped-Dataset/'

files = glob.glob(os.path.join(input_dir, "*.png"))

for file in files:
    filename = os.path.basename(file)
    image = Image.open(file)
    resized_image = image.resize((640, 480), Image.Resampling.BILINEAR)
    resized_image.save(f"{output_dir}{filename}")
    print(f"{output_dir}{filename}")
