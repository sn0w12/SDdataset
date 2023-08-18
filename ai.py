import os
import collections
import re
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import subprocess
import shutil
import platform
import sys
from send2trash import send2trash
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import tkinter as tk
from tkinter import filedialog

# Function to rename .safetensor files
def rename_safetensor_files(directory, prefix):
    counter = 1
    for filename in os.listdir(directory):
        if filename.endswith('.safetensors'):
            new_filename = f'{prefix}-{counter}.safetensors'
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            counter += 1

# Function to analyze and process a dataset of images
def process_dataset(images_folder, similarity_threshold=0.985):
    def clear_output():
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

    root_dir = os.getcwd()

    os.chdir(root_dir)
    model_name = "clip-vit-base32-torch"
    supported_types = (".png", ".jpg", ".jpeg")
    supported_video_types = (".mp4", ".avi", ".mov")

    videos = [f for f in os.listdir(images_folder) if f.lower().endswith(supported_video_types)]
    if videos:
        answer = input("Videos found in folder, delete them? (Y/n)\n")
        if answer.lower() == "y":
            for video in videos:
                send2trash(os.path.join(images_folder, video))
            print("✅ Videos deleted.")
        else:
            print("❌ Not deleting videos, exiting...")
            sys.exit()

    img_count = len(os.listdir(images_folder))
    batch_size = min(250, img_count)

    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    if img_count == 0:
        print(f"💥 Error: No images found in {images_folder}")
    else:
        print("\n💿 Analyzing dataset...\n")
        dataset = fo.Dataset.from_dir(images_folder, dataset_type=fo.types.ImageDirectory)
        model = foz.load_zoo_model(model_name)
        embeddings = dataset.compute_embeddings(model, batch_size=batch_size)

        batch_embeddings = np.array_split(embeddings, batch_size)
        similarity_matrices = []
        max_size_x = max(array.shape[0] for array in batch_embeddings)
        max_size_y = max(array.shape[1] for array in batch_embeddings)

        for i, batch_embedding in enumerate(batch_embeddings):
            similarity = cosine_similarity(batch_embedding)
            # Pad 0 for np.concatenate
            padded_array = np.zeros((max_size_x, max_size_y))
            padded_array[0:similarity.shape[0], 0:similarity.shape[1]] = similarity
            similarity_matrices.append(padded_array)

        similarity_matrix = np.concatenate(similarity_matrices, axis=0)
        similarity_matrix = similarity_matrix[0:embeddings.shape[0], 0:embeddings.shape[0]]

        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix -= np.identity(len(similarity_matrix))

        dataset.match(F("max_similarity") > similarity_threshold)
        dataset.tags = ["delete", "has_duplicates"]

        id_map = [s.id for s in dataset.select_fields(["id"])]
        samples_to_remove = set()
        samples_to_keep = set()

        for idx, sample in enumerate(dataset):
            if sample.id not in samples_to_remove:
                # Keep the first instance of two duplicates
                samples_to_keep.add(sample.id)

                dup_idxs = np.where(similarity_matrix[idx] > similarity_threshold)[0]
                for dup in dup_idxs:
                    # We kept the first instance so remove all other duplicates
                    samples_to_remove.add(id_map[dup])

                if len(dup_idxs) > 0:
                    sample.tags.append("has_duplicates")
                    sample.save()
            else:
                sample.tags.append("delete")
                sample.save()

        clear_output()

        sidebar_groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)
        for group in sidebar_groups[1:]:
            group.expanded = False
        dataset.app_config.sidebar_groups = sidebar_groups
        dataset.save()
        clear_output()
        print("💾 Saving...")

        kys = [s for s in dataset if "delete" in s.tags]
        dataset.delete_samples(kys)
        project_subfolder = "output"  # Choose a project subfolder name to export the dataset
        previous_folder = images_folder[:images_folder.rfind("\\")]
        dataset.export(export_dir=os.path.join(images_folder, project_subfolder), dataset_type=fo.types.ImageDirectory)

        temp_suffix = "_temp"
        shutil.move(images_folder, images_folder + temp_suffix)
        shutil.move(images_folder + temp_suffix + "\\" + project_subfolder, images_folder)
        shutil.rmtree(images_folder + temp_suffix)

        print(f"\n✅ Removed {len(kys)} images from the dataset. You now have {len(os.listdir(images_folder))} images.")

# Function to copy image files from a source folder to a destination folder
def copy_images(source_folder, destination_folder):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Destination folder '{destination_folder}' created.")
    
    # Get all files in the source folder
    files = os.listdir(source_folder)
    
    # Copy each image file to the destination folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    num_copied = 0
    for file in files:
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in image_extensions:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copyfile(source_path, destination_path)
            num_copied += 1
    
    print(f"{num_copied} images copied to '{destination_folder}'.")

# Function to count items in a text file and rank them
def count_items(file_path, item_count):
    with open(file_path, 'r') as file:
        # Use a regex pattern to match items separated by commas or newlines
        pattern = r'[^,\n]+'
        text = file.read()
        items = re.findall(pattern, text)
        for item in items:
            item_count[item] += 1

def rank_items(folder_path):
    item_count = collections.defaultdict(int)
    if os.path.isfile(folder_path):
        folder_path = os.path.dirname(folder_path)

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                count_items(file_path, item_count)
    
    ranked_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)[:30]  # Display only the top 30 items
    return ranked_items

# Function to convert an image to black and white
def convert_to_bw(image_path):
    image = Image.open(image_path)
    bw_image = image.convert("L")  # Convert to grayscale
    bw_image.save(image_path)

def convert_folder_to_bw(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over each file in the folder
    for file_name in file_list:
        # Check if the file is an image
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # Get the full path of the image
            image_path = os.path.join(folder_path, file_name)

            # Convert the image to black and white
            convert_to_bw(image_path)

    print("Conversion complete!")

# Function to organize files based on a keyword in text files
def create_subfolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_and_copy_files(root_folder, keyword):
    for root, _, files in os.walk(root_folder):
        for filename in files:
            name, extension = os.path.splitext(filename)
            if extension == '.txt':
                with open(os.path.join(root, filename), 'r') as file:
                    content = file.read()
                    if keyword in content:
                        subfolder_name = keyword.lower()
                        create_subfolder(subfolder_name)

                        # Create copies of the text file
                        copy_txt_path = os.path.join(subfolder_name, filename)
                        shutil.copy(os.path.join(root, filename), copy_txt_path)

                        # Check if the image file exists in the same directory and copy it if it does
                        image_file = name + '.jpg'
                        image_file_path = os.path.join(root, image_file)
                        if os.path.exists(image_file_path):
                            copy_img_path = os.path.join(subfolder_name, image_file)
                            shutil.copy(image_file_path, copy_img_path)

                        print(f"Copied '{filename}' and '{image_file}' to '{subfolder_name}' subfolder.")

# Function to check if an image has transparent pixels
def has_transparent_pixels(image):
    if image.mode in ('RGBA', 'LA'):
        alpha_channel = image.split()[-1]
        return any(alpha_channel.getdata())
    return False

def replace_transparent_with_white(image_path):
    try:
        image = Image.open(image_path)
        if has_transparent_pixels(image):
            # Resize the image to a smaller resolution
            width, height = image.size
            max_size = 1000  # You can adjust this value based on your requirements
            if width > max_size or height > max_size:
                image.thumbnail((max_size, max_size))

            # Create a white background
            white_background = Image.new('RGB', image.size, (255, 255, 255))

            # Composite the image on the white background, ignoring alpha
            white_image = Image.alpha_composite(white_background.convert('RGBA'), image)

            # Convert to RGB mode (removing the alpha channel)
            white_image = white_image.convert('RGB')

            # Save the modified image
            white_image.save(image_path)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_images_in_folder(folder_path, batch_size=10):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    # Process images in batches with a progress bar
    num_images = len(image_paths)
    with tqdm(total=num_images, desc="Processing images", unit="image") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(0, num_images, batch_size):
                batch = image_paths[i:i + batch_size]
                list(executor.map(replace_transparent_with_white, batch))
                pbar.update(len(batch))

def process_dataset_and_replace_transparent(images_folder, similarity_threshold=0.985):
    process_dataset(images_folder, similarity_threshold)
    process_images_in_folder(images_folder)
    
def panel_extraction():
    # Open a directory dialog for selecting the image directory
    dir_path = filedialog.askdirectory(title="Select a Directory")

    if not dir_path:
        print("No directory selected.")
        exit()
    # Load all images from the selected directory
    image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # Initialize a counter for saved panels
    total_saved_panels = 0

    # Minimum panel area threshold (adjust as needed)
    min_panel_area = 0.1  # For example, 1% of the total image area
    # Process the images twice
    for iteration in range(1):
        print(f"Running iteration {iteration + 1}...")

        # Process each image
        for image_file in tqdm(image_files, desc="Processing Images", unit="image"):
            image_path = os.path.join(dir_path, image_file)
            im = Image.open(image_path).convert('L')
            im_array = np.array(im)

            plt.figure(figsize=(18, 12))

            gs = plt.GridSpec(2, 4)

            plt.subplot(gs[0, 0])
            plt.imshow(im_array, cmap='gray')
            plt.title("Original Grayscale Image")

            # Apply Gaussian blur
            blurred_image = gaussian(im_array, sigma=2.5)

            plt.subplot(2, 3, 2)
            plt.imshow(blurred_image, cmap='gray')
            plt.title("Blurred Image")

            # Apply edge detection
            edges = canny(blurred_image)

            plt.subplot(2, 3, 3)
            plt.imshow(edges, cmap='gray')
            plt.title("Edge Detection")

            # Enhance edges
            thick_edges = dilation(dilation(edges))

            plt.subplot(2, 3, 4)
            plt.imshow(thick_edges, cmap='gray')
            plt.title("Enhanced Edges")

            # Fill holes in the edges
            segmentation = ndi.binary_fill_holes(thick_edges)

            # Erode the segmentation to remove small lines and artifacts
            num_erosion_iterations = 3  # Increase this value for more erosion
            eroded_segmentation = ndi.binary_erosion(segmentation, iterations=num_erosion_iterations)

            # Dilate the eroded segmentation back to the original size
            dilated_segmentation = ndi.binary_dilation(eroded_segmentation, iterations=2)

            # Label connected components
            labels = label(dilated_segmentation)

            plt.subplot(2, 3, 6)
            plt.imshow(label2rgb(labels, bg_label=0))
            plt.title("Labeled Regions")

            # Define functions
            def do_bboxes_overlap(a, b):
                return (
                    a[0] < b[2] and
                    a[2] > b[0] and
                    a[1] < b[3] and
                    a[3] > b[1] and
                    ((a[0] + a[2]) / 2 < (b[0] + b[2]) / 2 or (a[1] + a[3]) / 2 < (b[1] + b[3]) / 2)
                )

            def merge_bboxes(a, b):
                return (
                    min(a[0], b[0]),
                    min(a[1], b[1]),
                    max(a[2], b[2]),
                    max(a[3], b[3])
                )

            # Identify panels
            regions = regionprops(labels)
            panels = []

            for region in regions:
                for i, panel in enumerate(panels):
                    if do_bboxes_overlap(region.bbox, panel):
                        panels[i] = merge_bboxes(panel, region.bbox)
                        break
                else:
                    panels.append(region.bbox)

            # Filter out small panels
            min_panel_area = 0.1  # For example, 1% of the total image area
            panels = [bbox for bbox in panels if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= min_panel_area * im_array.size]

            # Create 'panels' directory within the image directory
            panels_dir = os.path.join(dir_path, 'panels')
            os.makedirs(panels_dir, exist_ok=True)

            # Save individual panels
            num_saved_panels = 0  # Initialize a counter for saved panels in this image
            for i, bbox in enumerate(panels):
                panel = im_array[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                panel_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if panel_area >= min_panel_area * im_array.size and panel_area <= 0.9 * im_array.size:
                    panel_path = os.path.join(panels_dir, f'{i}_{image_file}')
                    panel_image = Image.fromarray(panel)
                    panel_image.save(panel_path)
                    num_saved_panels += 1
            
            # Update the total saved panels counter
            total_saved_panels += num_saved_panels
            
            # Close the figures to free up memory
            plt.close('all')
        
    # Print the total number of saved panels
    print(f"Total {total_saved_panels} panels saved across {len(image_files)} images.")

def sort_lora():
    # Function to search for files with a given name in a directory and its subdirectories
    def search_files(directory, name):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if name in file:
                    return os.path.join(root, file)
        return None

    # Paths to the source CSV file and destination folders
    csv_file_path = r"F:\stable-diffusion-webui\styles.csv"
    lora_destination_folder = r"C:\Users\lucas\ai\Lora"
    lycoris_destination_folder = r"C:\Users\lucas\ai\LyCORIS"

    # Read the CSV file using pandas
    df = pd.read_csv(csv_file_path, skiprows=1)

    # Initialize lists to store the required text patterns
    lora_patterns = []
    lycoris_patterns = []

    # Extract patterns from the entire CSV file
    for _, row in df.iterrows():
        for pattern in row.dropna().tolist():
            lora_matches = re.findall(r"<lora:([^:>]+):[0-9.]+>", pattern)
            lora_patterns.extend(lora_matches)

            lycoris_matches = re.findall(r"<lyco:([^:>]+):[0-9.]+>", pattern)
            lycoris_patterns.extend(lycoris_matches)

    # Create progress bars for both sets of patterns
    lora_progress_bar = tqdm(lora_patterns, desc="Searching Lora patterns", unit="file")
    lycoris_progress_bar = tqdm(lycoris_patterns, desc="Searching LyCORIS patterns", unit="file")

    # Search for and copy files with the found names to the destination folders
    for pattern in lora_progress_bar:
        file_path = search_files(r"F:\stable-diffusion-webui", pattern)
        if file_path:
            shutil.copy(file_path, os.path.join(lora_destination_folder, os.path.basename(file_path)))
        lora_progress_bar.update()

    for pattern in lycoris_progress_bar:
        file_path = search_files(r"F:\stable-diffusion-webui", pattern)
        if file_path:
            shutil.copy(file_path, os.path.join(lycoris_destination_folder, os.path.basename(file_path)))
        lycoris_progress_bar.update()
        
def delete_files_without_words(folder_path):
    # List of words that should be present in the txt files
    words_to_check = ['1girl', '1boy']
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            txt_path = os.path.join(folder_path, filename)
            image_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.jpg")

            with open(txt_path, 'r') as txt_file:
                content = txt_file.read()
                
            if all(word not in content for word in words_to_check):
                os.remove(txt_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
                print(f"Deleted {filename} and its corresponding image.")

# Main menu to choose the features
def main_menu():
    images_folder = None

    while True:
        print("Main Menu:")
        print("1. Rename .safetensor files")
        print("2. Analyze and delete duplicate images")
        print("3. Copy images from a source folder to a destination folder")
        print("4. Rank tags in text files")
        print("5. Convert images to black and white")
        print("6. Organize files based on a tag in text files")
        print("7. Replace transparent pixels with white in images")
        print("8. Process dataset and replace transparent pixels in images")
        print("9. Copy all loras and lycoris files that are in styles.csv to seperate folders")
        print("10. Try to seperate a manga page into individual panels")
        print("11. Remove all images and textfiles without 1girl or 1boy in them")
        print("0. Exit")

        choice = input("Enter your choice (1/2/3/4/5/6/7/8/9/10/11/0): ")

        if choice == "1":
            # Rename .safetensor files
            directory = input("Enter the directory path where the .safetensor files are located: ")
            if not os.path.isdir(directory):
                print("Invalid directory path.")
            else:
                prefix = input("Enter the prefix for renaming: ")
                rename_safetensor_files(directory, prefix)
        elif choice == "2":
            # Analyze and process a dataset of images
            images_folder = input("Enter the path to the image folder: ")
            process_dataset(images_folder)
        elif choice == "3":
            # Copy images from a source folder to a destination folder
            source_folder = input("Enter the source folder path: ")
            destination_folder = input("Enter the destination folder path: ")
            copy_images(source_folder, destination_folder)
        elif choice == "4":
            # Rank items in text files
            folder_path = input("Enter the folder path: ")
            ranked_items = rank_items(folder_path)
            for item, count in ranked_items:
                print(f'{item.strip()}: {count}')
        elif choice == "5":
            # Convert images to black and white
            folder_path = input("Enter the path to the folder containing the images: ")
            convert_folder_to_bw(folder_path)
        elif choice == "6":
            # Organize files based on a keyword in text files
            root_folder = input("Enter the path of the folder containing images and text files: ")
            keyword = input("Enter the keyword to search for in the text files: ")
            find_and_copy_files(root_folder, keyword)
        elif choice == "7":
            # Replace transparent pixels with white in images
            images_folder = input("Enter the path to the folder containing images: ")
            process_images_in_folder(images_folder)
        elif choice == "8":
            # Process dataset and replace transparent pixels in images
            images_folder = input("Enter the path to the folder containing images: ")
            process_dataset_and_replace_transparent(images_folder)
        elif choice == "9":
            sort_lora()
        elif choice == "10":
            panel_extraction()
        elif choice == "11":
            images_folder = input("Enter the path to the folder containing images and text files: ")
            delete_files_without_words(folder_path)
        elif choice == "0":
            # Exit the program
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
