import os
import collections
import re
from PIL import Image, ImageTk, ImageDraw
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
from tkinter import filedialog, ttk
import cv2
import random
import string
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from skimage import img_as_ubyte
import glob

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


def get_top_items(folder_path):
    item_count = collections.defaultdict(int)
    if os.path.isfile(folder_path):
        folder_path = os.path.dirname(folder_path)

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                count_items(file_path, item_count)

    # Sort items by count in descending order
    sorted_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)

    # Determine the highest count
    highest_count = sorted_items[0][1] if sorted_items else 0

    # Filter items to include only those with count >= 10% of the highest count
    top_items = [item for item in sorted_items if item[1] >= 0.1 * highest_count]

    return top_items

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

def process_and_update(image_file, dir_path, panels_dir, min_panel_area, index, total_images):
    image_path = os.path.join(dir_path, image_file)
    saved_panels, processed_images = process_image(image_path, dir_path, panels_dir, min_panel_area)

    return saved_panels, processed_images, image_file

def display_images(processed_images_data, scrollable_frame):
    for image_data in processed_images_data:
        image_frame = ttk.Frame(scrollable_frame)
        image_frame.pack(fill="x", expand=True)

        for img_array, title in image_data['images']:
            if (title == "Original Image"):
                add_image_to_gui(img_array, f"{image_data['file']}", image_frame)
            else:
                add_image_to_gui(img_array, f"{title}", image_frame)

def panel_extraction():
    root = tk.Tk()
    root.title("Image Processing Steps")
    root.state('zoomed')  # For Windows

    # Set an initial size for the window
    num_steps = 7  # Number of processing steps (including the original image)
    initial_window_width = 350 * num_steps  # 350 pixels per step
    initial_window_height = 400  # Enough to display images and titles
    root.geometry(f"{initial_window_width}x{initial_window_height}")

    # Create a canvas and scrollbar for scrolling images
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    dir_path = filedialog.askdirectory(title="Select a Directory")
    if not dir_path:
        print("No directory selected.")
        exit()

    image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    total_images = len(image_files)

    total_saved_panels = 0
    min_panel_area = 0.05
    panels_dir = os.path.join(dir_path, 'panels')
    os.makedirs(panels_dir, exist_ok=True)

    processed_images_data = []

    with ThreadPoolExecutor() as executor:
        # Wrap your tasks in tqdm to create a progress bar
        futures = {executor.submit(process_and_update, image_file, dir_path, panels_dir, min_panel_area, index, total_images): image_file 
                for index, image_file in enumerate(image_files)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(image_files), desc="Processing Images"):
            saved_panels, processed_images, image_file = future.result()
            processed_images_data.append({'file': image_file, 'images': processed_images})
            total_saved_panels += saved_panels
            root.after(0)

    # Display images after all processing is done
    display_images(processed_images_data, scrollable_frame)

    print(f"Total {total_saved_panels} panels saved across {total_images} images.")
    root.mainloop()

def add_image_to_gui(img_array, title, parent_frame):
    max_image_size = (350, 350)  # Adjust size as needed
    img = Image.fromarray((img_array * 255).astype(np.uint8) if img_array.dtype == np.bool_ else img_array)
    img.thumbnail(max_image_size, Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    frame = ttk.Frame(parent_frame)
    frame.pack(side=tk.LEFT, padx=5, pady=5)
    label = ttk.Label(frame, image=imgtk)
    label.image = imgtk  # Keep a reference
    label.pack()
    title_label = ttk.Label(frame, text=title)
    title_label.pack()

def draw_and_save_bounding_boxes(im_array, panels, panels_dir, image_file):
    # Define a list of 10 colors in RGBA format (with alpha for transparency)
    colors = [
        (255, 0, 0, 128),   # Red with 50% opacity
        (0, 255, 0, 128),   # Green with 50% opacity
        (0, 0, 255, 128),   # Blue with 50% opacity
        (255, 255, 0, 128), # Yellow with 50% opacity
        (255, 0, 255, 128), # Magenta with 50% opacity
        (0, 255, 255, 128), # Cyan with 50% opacity
        (255, 165, 0, 128), # Orange with 50% opacity
        (255, 105, 180, 128), # Pink with 50% opacity
        (105, 255, 180, 128), # Mint with 50% opacity
        (255, 255, 255, 128), # White with 50% opacity
    ]

    img_for_boxes = Image.fromarray(im_array).convert('RGBA')
    overlay = Image.new('RGBA', img_for_boxes.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for i, bbox in enumerate(panels):
        color = colors[i % len(colors)]
        draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline=color, fill=color)
    
    img_for_boxes = Image.alpha_composite(img_for_boxes, overlay)
    img_for_boxes = img_for_boxes.convert('RGB')

    # Instead of adding the image to the GUI, return the processed image array
    return np.array(img_for_boxes)

def process_image(image_path, dir_path, panels_dir, min_panel_area):
    # Open the image and convert to grayscale
    im = Image.open(image_path).convert('L')
    im_array = np.array(im)

    processed_images = []

    # Include the original image
    processed_images.append((im_array, "Original Image"))

    # Apply Gaussian blur
    blurred_image = gaussian(im_array, sigma=2.5)
    # Convert the blurred image to an 8-bit format
    blurred_image_ubyte = img_as_ubyte(blurred_image)
    processed_images.append((blurred_image_ubyte, "Blurred Image"))

    # Apply edge detection
    edges = canny(blurred_image)
    processed_images.append((edges, "Edge Detection"))

    # Enhance edges
    thick_edges = dilation(dilation(edges))
    processed_images.append((thick_edges, "Thick Edges"))

    # Fill holes in the edges
    segmentation = ndi.binary_fill_holes(thick_edges)
    processed_images.append((segmentation, "Segmentation"))

    # Erode the segmentation to remove small lines and artifacts
    num_erosion_iterations = 12  # Increase this value for more erosion
    eroded_segmentation = ndi.binary_erosion(segmentation, iterations=int(num_erosion_iterations))
    # processed_images.append((eroded_segmentation, "Eroded Segmentation"))

    # Dilate the eroded segmentation back to the original size
    dilated_segmentation = ndi.binary_dilation(eroded_segmentation, iterations=int(num_erosion_iterations * 0.5))
    processed_images.append((dilated_segmentation, "Dilated Segmentation"))

    # Label connected components
    labels = label(dilated_segmentation)

    # Identify panels
    regions = regionprops(labels)
    panels = [region.bbox for region in regions if (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]) >= min_panel_area * im_array.size]

    # Call the modified draw_and_save_bounding_boxes function
    bounding_box_image = draw_and_save_bounding_boxes(im_array, panels, panels_dir, os.path.basename(image_path))
    processed_images.append((bounding_box_image, "Bounding Boxes"))

    # Save individual panels and count them
    num_saved_panels = 0
    for i, bbox in enumerate(panels):
        panel = im_array[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        panel_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if min_panel_area * im_array.size <= panel_area <= 0.9 * im_array.size:
            panel_path = os.path.join(panels_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_{i}.png')
            panel_image = Image.fromarray(panel)
            panel_image.save(panel_path)
            num_saved_panels += 1

    # Return the number of saved panels and the processed images for display
    return num_saved_panels, processed_images

def sort_lora():
    def search_files(directory, name):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if name in file:
                    return os.path.join(root, file)
        return None

    csv_file_path = r"F:\Ai\stable-diffusion-webui\styles.csv"
    lora_destination_folder = r"C:\Users\lucas\ai\Lora"

    df = pd.read_csv(csv_file_path, skiprows=7)

    lora_patterns = []

    for _, row in df.iterrows():
        for pattern in row.dropna().tolist():
            if not isinstance(pattern, str):
                continue  # Skip non-string patterns
            lora_matches = re.findall(r"<lora:([^:>]+):[0-9.]+>", pattern)
            lora_patterns.extend(lora_matches)

    lora_progress_bar = tqdm(lora_patterns, desc="Searching Lora patterns", unit="file")

    for pattern in lora_progress_bar:
        file_path = search_files(r"F:\Ai\stable-diffusion-webui", pattern)
        if file_path:
            shutil.copy(file_path, os.path.join(lora_destination_folder, os.path.basename(file_path)))
        lora_progress_bar.update()
        
def delete_files_without_words(folder_path):
    # List of words that should be present in the txt files
    words_to_check = ['1girl', '1boy']
    for filename in tqdm(os.listdir(folder_path), desc="Checking files"):
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

def delete_files_with_words(folder_path, words_to_check):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            txt_path = os.path.join(folder_path, filename)
            image_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.jpg")

            with open(txt_path, 'r') as txt_file:
                content = txt_file.read()

            if any(word in content for word in words_to_check):
                os.remove(txt_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
                print(f"Deleted {filename} and its corresponding image.")

def generate_random_string(length=6):
    """Generate a random string of lowercase letters and digits."""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def save_frames(video_path, output_folder, interval=10):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate the time in seconds for the current frame
        current_time = frame_number / fps
        
        if current_time % interval == 0:
            # Generate a random filename for the frame
            random_name = generate_random_string()
            output_path = os.path.join(output_folder, f"frame_{random_name}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_number} at time {current_time} seconds as {output_path}")

        frame_number += 1

    cap.release()

def resize_images(folder_path, max_length):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Loop through each file
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file is an image
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            try:
                # Open the image file
                image = Image.open(file_path)
                
                # Get the original width and height
                width, height = image.size
                
                # Calculate the aspect ratio
                aspect_ratio = width / height
                
                # Resize the image based on the maximum length while preserving aspect ratio
                if width > height:
                    new_width = max_length
                    new_height = int(max_length / aspect_ratio)
                else:
                    new_height = max_length
                    new_width = int(max_length * aspect_ratio)
                
                # Resize the image
                resized_image = image.resize((new_width, new_height))
                
                # Save the resized image, overwriting the original file
                resized_image.save(file_path)
                
                print(f"Resized: {file_name}")
            
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

def process_image(input_image_path, output_folder_base, size_threshold=0.05):
    # Load the image
    image = Image.open(input_image_path)
    image_np = np.array(image)

    # Define a broader green color range
    lower_green = np.array([0, 200, 0])  # Adjust these values
    upper_green = np.array([10, 255, 10])  # Adjust these values

    # Create a mask to identify non-green areas
    mask = cv2.inRange(image_np, lower_green, upper_green)
    non_green_areas = cv2.bitwise_not(mask)  # Invert mask to get non-green areas

    # Find disconnected components (non-green areas)
    num_labels, labels = cv2.connectedComponents(non_green_areas)

    # Calculate the size threshold based on the original image size
    original_size = image_np.shape[0] * image_np.shape[1]
    min_size = original_size * size_threshold

    saved_images = 0

    # Iterate through the found components
    for label in range(1, num_labels):  # Start from 1 to ignore the background
        component_mask = np.where(labels == label, 255, 0).astype('uint8')
        x, y, w, h = cv2.boundingRect(component_mask)
        if w * h > min_size:
            cropped_image = image_np[y:y+h, x:x+w]
            cropped_mask = component_mask[y:y+h, x:x+w]
            cropped_image[cropped_mask == 0] = [255, 255, 255]
            unique_filename = f"{random_string()}.png"
            Image.fromarray(cropped_image).save(os.path.join(output_folder_base, unique_filename))

def remove_green_and_separate(input_folder, size_threshold=0.05):
    # Supported image formats
    formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    # Get a list of all images in the input folder of supported formats
    input_images = []
    for format in formats:
        input_images.extend(glob.glob(os.path.join(input_folder, format)))

    # Create the output folder within the same directory as the input folder
    output_folder_base = os.path.join(input_folder, "separated_parts")
    if not os.path.exists(output_folder_base):
        os.makedirs(output_folder_base)

    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        # Map the process_image function to each image
        list(tqdm(executor.map(process_image, input_images, [output_folder_base]*len(input_images), [size_threshold]*len(input_images)), total=len(input_images)))

def rename_txt_files(folder_path):
    for filename in tqdm(os.listdir(folder_path), desc="Renaming files"):
        # Check if the file is a .txt file and has the specified format
        if filename.endswith(".txt_01.txt"):
            # Construct the new filename
            new_filename = filename.replace("_01.txt", "")
            # Rename the file
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    print(f"Finished renaming textfiles")

def random_string(length=10):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            new_name = random_string() + os.path.splitext(filename)[1]
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
            print(f"Renamed '{filename}' to '{new_name}'")

def git_pull_in_subfolders(parent_directory):
    # Change the working directory to the specified parent directory
    os.chdir(parent_directory)

    # List all the entries in the directory
    for entry in os.listdir(parent_directory):
        path = os.path.join(parent_directory, entry)
        # Check if the entry is a directory
        if os.path.isdir(path):
            print(f"Entering {entry} and pulling...")
            # Change into the subdirectory
            os.chdir(path)
            try:
                # Run 'git pull' command
                subprocess.run(["git", "pull"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to pull in {entry}: {e}")
            # Change back to the parent directory
            os.chdir(parent_directory)

# Main menu to choose the features
def main_menu():
    directory = None

    while True:
        number = 1
        print("Main Menu:")
        print(str(number) + ". Rename .safetensor files"); number += 1
        print(str(number) + ". Analyze and delete duplicate images"); number += 1
        print(str(number) + ". Copy images from a source folder to a destination folder"); number += 1
        print(str(number) + ". Rank tags in text files"); number += 1
        print(str(number) + ". Convert images to black and white"); number += 1
        print(str(number) + ". Organize files based on a tag in text files"); number += 1
        print(str(number) + ". Replace transparent pixels with white in images"); number += 1
        print(str(number) + ". Process dataset and replace transparent pixels in images"); number += 1
        print(str(number) + ". Copy all loras and lycoris files that are in styles.csv to separate folders"); number += 1
        print(str(number) + ". Try to separate a manga page into individual panels"); number += 1
        print(str(number) + ". Remove all images and textfiles without 1girl or 1boy in them"); number += 1
        print(str(number) + ". Remove all images and textfiles with X tags in them"); number += 1
        print(str(number) + ". Save images from videos every X seconds"); number += 1
        print(str(number) + ". Resize images to X max length"); number += 1
        print(str(number) + ". Separate greenscreened images"); number += 1
        print(str(number) + ". Remove comfyui suffix from .txt file"); number += 1
        print(str(number) + ". Randomize file names"); number += 1
        print(str(number) + ". Git pull all subfolders"); number += 1
        print("0. Exit/Restart")

        allNumbers = ""
        for i in range(number - 1):
            allNumbers += (str(i + 1) + "/")
        choice = input("Enter your choice (" + allNumbers + "0): ").lower()

        if choice not in ["9", "10", "0"]:
            directory = input("Enter the directory path: ")
            if not os.path.isdir(directory):
                print("Invalid directory path.")
        match choice:
            case "1":
                # Rename .safetensor files
                    prefix = input("Enter the prefix for renaming: ")
                    rename_safetensor_files(directory, prefix)
            case "2":
                # Analyze and process a dataset of images
                process_dataset(directory)
            case "3":
                # Copy images from a source folder to a destination folder
                destination_folder = input("Enter the destination folder path: ")
                copy_images(directory, destination_folder)
            case "4":
                # Rank items in text files
                ranked_items = get_top_items(directory)  # Using the new get_top_items function
                for item, count in ranked_items:
                    if item.startswith(' '):
                        cleaned_item = item[1:]
                    else:
                        cleaned_item = item
                    print(f'{cleaned_item}: {count}')  # Diagnostic print

                prompt = ""
                counter = 0
                excluded_words = ["solo", "1girl", "looking at viewer", "smile", "cleavage", "hair ornament", "animal ears", "hairclip", "fake animal ears"] 

                print("\nPrompt:")
                for item in ranked_items:
                    cleaned_item = item[0][1:] if item[0].startswith(' ') else item[0]
                    if cleaned_item not in excluded_words:
                        prompt += cleaned_item + ", "
                        counter += 1
                        if counter == 20:
                            break
                
                print(prompt)

            case "5":
                # Convert images to black and white
                convert_folder_to_bw(directory)
            case "6":
                # Organize files based on a keyword in text files
                keyword = input("Enter the keyword to search for in the text files: ")
                find_and_copy_files(directory, keyword)
            case "7":
                # Replace transparent pixels with white in images
                process_images_in_folder(directory)
            case "8":
                # Process dataset and replace transparent pixels in images
                process_dataset_and_replace_transparent(directory)
            case "9":
                sort_lora()
            case "10":
                panel_extraction()
            case "11":
                delete_files_without_words(directory)
            case"12":
                words_input = input("Enter words (comma-separated) that should be present in the txt files: ")
                words_to_check = [word.strip() for word in words_input.split(",")]

                delete_files_with_words(directory, words_to_check)
            case "13":
                interval = int(input("Enter the time interval in seconds: "))
                output_folder = directory
                video_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]

                for video_file in video_files:
                    video_path = os.path.join(directory, video_file)
                    save_frames(video_path, output_folder, interval)
            case "14":
                max_length = int(input("Enter the max size of image: "))
                resize_images(directory, max_length)
            case "15":
                remove_green_and_separate(directory)
            case "16":
                rename_txt_files(directory)
            case "17":
                rename_files_in_folder(directory)
            case "18":
                git_pull_in_subfolders(directory)
            case "0":
                print("Exiting...")
                sys.exit()
            case _:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()