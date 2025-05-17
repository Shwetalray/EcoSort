import albumentations as A
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=20, p=0.5),
        A.GaussianBlur(p=0.2),
        A.HueSaturationValue(p=0.4),
        A.RandomShadow(p=0.3),
    ], additional_targets={'image': 'image'})

def augment_images(input_dir, output_dir, num_augmented=5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augment = get_augmentation_pipeline()

    for img_path in tqdm(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                logging.warning(f"Unable to read image: {img_path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(num_augmented):
                augmented = augment(image=image)['image']
                aug_name = f"{img_path.stem}_aug{i}.jpg"
                aug_path = output_dir / aug_name
                cv2.imwrite(str(aug_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

            # Copy original image as well
            original_path = output_dir / f"{img_path.stem}_original.jpg"
            cv2.imwrite(str(original_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")

def preprocess_images(input_dir, output_dir, resize=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(input_dir.glob("*.jpg")):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                logging.warning(f"Unable to read image: {img_path}")
                continue
            
            # Normalize the image
            image = image / 255.0  # Scale to [0, 1]

            # Resize if specified
            if resize:
                image = cv2.resize(image, resize)

            # Save the preprocessed image
            preprocessed_path = output_dir / img_path.name
            cv2.imwrite(str(preprocessed_path), (image * 255).astype(np.uint8))  # Convert back to uint8 for saving

        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")

def main():
    # Define your dataset paths
    augment_images("dataset/train/plastic", "dataset_augmented/train/plastic", num_augmented=5)
    augment_images("dataset/train/paper", "dataset_augmented/train/paper", num_augmented=5)

    # Preprocess the augmented images
    preprocess_images("dataset_augmented/train/plastic", "dataset_preprocessed/train/plastic", resize=(224, 224))
    preprocess_images("dataset_augmented/train/paper", "dataset_preprocessed/train/paper", resize=(224, 224))

    # Preprocess validation images if they exist
    if os.path.exists("dataset/val/plastic"):
        preprocess_images("dataset/val/plastic", "dataset_preprocessed/val/plastic", resize=(224, 224))
    if os.path.exists("dataset/val/paper"):
        preprocess_images("dataset/val/paper", "dataset_preprocessed/val/paper", resize=(224, 224))

if __name__ == "__main__":
    main() 