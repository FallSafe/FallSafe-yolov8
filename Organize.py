import os
import shutil
from tqdm import tqdm

# Define paths
dataset_path = 'path_to_roboflow_extracted_folder'
subsets = ['train', 'valid', 'test']

# Define your classes (example: ['class1', 'class2'])
classes = ['fall', 'nofall']

# Function to create class directories
def create_class_directories(subset_path):
    for cls in classes:
        class_dir = os.path.join(subset_path, cls)
        os.makedirs(class_dir, exist_ok=True)

# Function to move images based on labels
def move_images(subset_path):
    images_path = os.path.join(subset_path, 'images')
    labels_path = os.path.join(subset_path, 'labels')

    label_files = os.listdir(labels_path)
    for label_file in tqdm(label_files, desc=f"Processing {subset_path}"):
        label_path = os.path.join(labels_path, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue
            class_id = int(lines[0].strip().split()[0])
            class_name = classes[class_id]

            image_file = label_file.replace('.txt', '.jpg')  # Assuming images are .jpg
            image_path = os.path.join(images_path, image_file)
            if os.path.exists(image_path):
                dst_path = os.path.join(subset_path, class_name, image_file)
                shutil.move(image_path, dst_path)

# Process each subset
for subset in subsets:
    subset_path = os.path.join(dataset_path, subset)
    create_class_directories(subset_path)
    move_images(subset_path)

print("Images have been moved to respective class directories.")
