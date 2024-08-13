import glob
import os
import cv2
import math
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from .network import ConvConvNext, FCLayerConvNext


def random_rotation(image, image_type, rotation_angle):
    """
    Rotate the image and fill empty pixels with a pixel value 192 if it's not a mask.

    Args:
    - image (numpy.ndarray or PIL.Image): Input image.
    - image_type (str): Type of the image ("mask" or "image").
    - rotation_angle (float): Angle of rotation in degrees.

    Returns:
    - rotated_array (numpy.ndarray): Rotated image array.
    """

    # Convert input to Pillow Image if it's a NumPy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Rotate the image
    rotated_array = np.array(image.rotate(rotation_angle, resample=Image.Resampling.BILINEAR))

    # Check if the image is a mask
    if image_type.lower() != "mask":
        # If it's not a mask, fill empty pixels with 192
        rotated_array[rotated_array == 0] = 192

    return rotated_array


def find_pupil_circle_using_convnext(circle_model_path, image, input_transform, device):
    """
    Find pupil circle using ConvNext model.

    Args:
    - circle_model_path (str): Path to the trained ConvNet model.
    - image (np.array): Input image as numpy array.
    - input_transform (torchvision.transforms): Input transformation for the model.
    - device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
    - pupil_x (float): X-coordinate of the pupil center.
    - pupil_y (float): Y-coordinate of the pupil center.
    - pupil_r (float): Radius of the pupil circle.
    """

    # Convert numpy array to PIL Image
    image = Image.fromarray(image)

    # Get image width and height
    w, h = image.size

    # Load ConvNet model
    circle_model = models.convnext_tiny()
    circle_model.avgpool = ConvConvNext(in_channels=768, out_n=6)
    circle_model.classifier = FCLayerConvNext(in_h=7, in_w=10, out_n=6)

    try:
        # Load model weights from file
        circle_model.load_state_dict(torch.load(circle_model_path, map_location=device))
    except AssertionError:
        print("Assertion error occurred while loading model weights.")

    # Move model to specified device (CPU or GPU)
    circle_model = circle_model.to(device)

    # Set model to evaluation mode
    circle_model.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Apply input transformation, unsqueeze to add batch dimension, and move to device
        inp_xyr_t = circle_model(Variable(input_transform(image).unsqueeze(0).repeat(1, 3, 1, 1).to(device)))

    # Circle parameters
    diag = math.sqrt(w ** 2 + h ** 2)
    inp_xyr = inp_xyr_t.tolist()[0]
    pupil_x = (inp_xyr[0] * w)
    pupil_y = (inp_xyr[1] * h)
    pupil_r = (inp_xyr[2] * 0.5 * 0.8 * diag)
    # iris_x = (inp_xyr[3] * w)
    # iris_y = (inp_xyr[4] * h)
    # iris_r = (inp_xyr[5] * 0.5 * diag)

    return pupil_x, pupil_y, pupil_r


def find_pupil_circle_using_hough(image, mask):
    """
    Find the coordinates and radius of the pupil using Hough Circle Transform.

    Args:
    - image (numpy.ndarray): Input image.
    - mask (numpy.ndarray): Mask representing the pupil area.

    Returns:
    - pupil_x (int): X-coordinate of the pupil center.
    - pupil_y (int): Y-coordinate of the pupil center.
    - pupil_r (int): Radius of the pupil circle.
    """

    # Check if input image or mask is None
    if image is None or mask is None:
        print("Error: Input image or mask is None.")
        return None, None, None

    # Create a mask for iris
    mask_for_iris = cv2.bitwise_not(mask)

    # Find non-zero indices in the mask
    iris_indices = np.nonzero(mask)

    if len(iris_indices[0]) == 0:
        return None, None, None

    # Compute spans in x and y directions
    y_span = max(iris_indices[0]) - min(iris_indices[0])
    x_span = max(iris_indices[1]) - min(iris_indices[1])

    # Estimate iris radius
    iris_radius_estimate = np.max((x_span, y_span)) // 2

    # Detect circles for iris
    iris_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                   param1=30,
                                   param2=5,
                                   minRadius=iris_radius_estimate - 32,
                                   maxRadius=iris_radius_estimate + 32)

    # Check if iris circle is detected
    if iris_circle is not None:
        iris_x, iris_y, iris_r = np.round(np.array(iris_circle[0][0])).astype(int)

        # Parameters for detecting pupil circle
        pupil_hough_param1 = 30
        pupil_hough_param2 = 5
        pupil_hough_minimum = 8
        pupil_iris_max_ratio = 0.7
        max_pupil_iris_shift = 25

        # Detect circles for pupil
        pupil_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                        param1=pupil_hough_param1,
                                        param2=pupil_hough_param2,
                                        minRadius=pupil_hough_minimum,
                                        maxRadius=np.int32(pupil_iris_max_ratio * iris_r))

        # Check if pupil circle is detected
        if pupil_circle is not None:
            # Extract pupil coordinates and radius
            pupil_x, pupil_y, pupil_r = np.round(np.array(pupil_circle[0][0])).astype(int)

            # Check the shift condition
            if np.sqrt((pupil_x - iris_x) ** 2 + (pupil_y - iris_y) ** 2) > max_pupil_iris_shift:
                pupil_x = iris_x
                pupil_y = iris_y
                pupil_r = iris_r // 3

            return pupil_x, pupil_y, pupil_r

    return None, None, None


def adjust_pupil_color(image, mask, input_transform, circle_model_path, circle_model_name,
                       pupil_pixel_range, aug_num_repetitions, device):
    """
    Adjust pupil color in the image using either a convolutional neural network model or Hough circle detection.

    Args:
    - circle_model_path (str): Path to the circle detection model (only used if circle_model_name is 'convnext').
    - image (numpy.ndarray): Input image.
    - mask (numpy.ndarray): Input image mask.
    - input_transform (function): Transformation function for input images.
    - circle_model_name (str): Name of the circle detection model to use ('convnext' or 'hough').
    - pupil_pixel_range (tuple): Range of pixel values for color adjustment.
    - aug_num_repetitions (int): Number of times to repeat the augmentation process for each input data.
    - device (torch.device): Device to use for running the model (only used if circle_model_name is 'convnext').

    Returns:
    - augmented_images (list): List of augmented images.
    - augmented_masks (list): List of corresponding augmented masks.
    """

    # Check if input image or mask is None
    if image is None or mask is None:
        print("Error: Input image or mask is None.")
        return None, None

    # Initialize lists to store augmented images and masks
    augmented_images = []
    augmented_masks = []

    # Find the coordinates and radius of the pupil
    if circle_model_name.lower() == 'convnext':
        pupil_x, pupil_y, pupil_r = find_pupil_circle_using_convnext(circle_model_path, image,
                                                                     input_transform, device)
    else:
        pupil_x, pupil_y, pupil_r = find_pupil_circle_using_hough(image, mask)

    # Create a pupil mask based on the detected circle
    if pupil_x is not None and pupil_y is not None and pupil_r is not None:
        x, y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
        pupil_mask = (x - pupil_x) ** 2 + (y - pupil_y) ** 2 <= pupil_r ** 2

        augmented_images.append(image)
        augmented_masks.append(mask)

        # Adjust pupil pixel color multiple times
        for i in range(aug_num_repetitions-1):
            # Adjust the pixel value of the pupil area in the image
            augmented_img = image.copy()
            augmented_img[pupil_mask] = np.random.randint(pupil_pixel_range[0], pupil_pixel_range[1])

            augmented_images.append(augmented_img)
            augmented_masks.append(mask)

    else:
        # If pupil is not detected, use original image and mask
        for i in range(aug_num_repetitions):
            augmented_images.append(image)
            augmented_masks.append(mask)

    return augmented_images, augmented_masks


class IrisDataset(Dataset):
    def __init__(self, image_dir, mask_dir, input_transform=None, target_transform=None,
                 mode='train', aug_num_repetitions=None, pupil_pixel_range=(109, 190),
                 circle_model_path=None, circle_model_name=None, device='cuda'):
        """
        Custom dataset for iris image segmentation.

        Args:
        - image_dir (str): Path to the directory containing input images.
        - mask_dir (str): Path to the directory containing masks.
        - input_transform (callable): Transformation function for input images.
        - target_transform (callable): Transformation function for target masks.
        - mode (str): Mode of the dataset ('train', 'val', or 'test').
        - aug_num_repetitions (int): Number of times to repeat the augmentation process for each input data.
        - pupil_pixel_range (tuple): Range of pixel values for color adjustment.
        - circle_model_path (str): Path to the circle detection model (only used if circle_model_name is 'convnext').
        - circle_model_name (str): Name of the circle detection model to use ('convnext' or 'hough').
        - device (str): Device to use for running the model (only used if circle_model_name is 'convnext').
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_transform = input_transform if input_transform is not None else ToTensor()
        self.target_transform = target_transform if target_transform is not None else ToTensor()
        self.mode = mode
        self.aug_num_repetitions = aug_num_repetitions
        self.pupil_pixel_range = pupil_pixel_range
        self.circle_model_path = circle_model_path
        self.circle_model_name = circle_model_name
        self.device = device

        # Collect image paths
        if self.image_dir[-1] == '/':
            self.all_image_paths = glob.glob(self.image_dir + '*.*')
        else:
            self.all_image_paths = glob.glob(os.path.join(self.image_dir, '*.*'))

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        """Load and preprocess an image and its mask."""
        # Get each image and mask path
        filename = os.path.basename(self.all_image_paths[idx])
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Load image and mask
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Resize image and mask
        image = image.resize((320, 240), Image.Resampling.BILINEAR)
        mask = mask.resize((320, 240), Image.Resampling.NEAREST)

        if self.aug_num_repetitions is not None and (self.mode == 'train' or self.mode == 'vis'):
            """ Will return 5D tensor """
            # Apply pupil color augmentation
            augmented_images, augmented_masks = adjust_pupil_color(np.array(image),
                                                                    np.array(mask),
                                                                    self.input_transform,
                                                                    self.circle_model_path,
                                                                    self.circle_model_name,
                                                                    self.pupil_pixel_range,
                                                                    self.aug_num_repetitions,
                                                                    self.device)

            # Apply random rotation to each image and its corresponding mask
            rotated_images = []
            rotated_masks = []
            for img, msk in zip(augmented_images, augmented_masks):
                rotation_angle = np.random.randint(-15, 15)
                rotated_images.append(random_rotation(img, "image", rotation_angle))
                rotated_masks.append(random_rotation(msk, "mask", rotation_angle))

            # Apply transformations
            image = torch.stack([self.input_transform(Image.fromarray(img)) for img in rotated_images])
            mask = torch.stack([self.target_transform(Image.fromarray(mask)) for mask in rotated_masks])

            return image, mask
        else:
            """ Will return 4D tensor """
            # Apply transformations
            image = self.input_transform(image)
            mask = self.target_transform(mask)

            return image, mask

