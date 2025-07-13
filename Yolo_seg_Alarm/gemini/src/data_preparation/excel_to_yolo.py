import pandas as pd
import os
from tqdm import tqdm

# --- Configuration ---
# TODO: Update these paths according to your project structure
EXCEL_FILE_PATH = 'd:/program/python/PythonProject/Yolo_seg_Alarm/data/raw/annotations.xlsx' # Path to your annotation file
IMAGE_DIR = 'd:/program/python/PythonProject/Yolo_seg_Alarm/data/raw/images' # Directory containing your images
OUTPUT_DIR = 'd:/program/python/PythonProject/Yolo_seg_Alarm/data/processed/labels' # Directory to save YOLO format labels

# TODO: Define your class names and their corresponding integer IDs
# The order matters. It should match the order used for training.
CLASS_MAPPING = {
    'excavator': 0,
    'piling_machine': 1,
    'pipe_jacking_machine': 2,
    # Add other classes if you have them
}

# TODO: Update these column names to match your Excel file
IMAGE_FILENAME_COL = 'image_name' # Column name for the image filename
CLASS_NAME_COL = 'category' # Column name for the class label
BBOX_COL = 'bbox_coco' # Column name for the bounding box coordinates

def convert_bbox_to_yolo(img_width, img_height, bbox):
    """
    Converts a bounding box from [xmin, ymin, w, h] format to YOLO's
    [x_center_norm, y_center_norm, width_norm, height_norm] format.

    Args:
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        bbox (list[float]): A list containing [xmin, ymin, w, h].

    Returns:
        list[float]: A list containing the bounding box in YOLO format.
    """
    xmin, ymin, w, h = bbox
    
    x_center = xmin + w / 2
    y_center = ymin + h / 2

    # Normalize coordinates
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = w / img_width
    height_norm = h / img_height

    return [x_center_norm, y_center_norm, width_norm, height_norm]

def main():
    """
    Main function to read annotations from an Excel file, convert them to
    YOLO format, and save them as .txt files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load the Excel file
    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {EXCEL_FILE_PATH}")
        print("Please update the EXCEL_FILE_PATH variable in the script.")
        return

    # Group annotations by image filename
    grouped = df.groupby(IMAGE_FILENAME_COL)

    print(f"Found {len(grouped)} unique images to process.")

    for filename, group in tqdm(grouped, desc="Processing images"):
        # Get image dimensions
        # This requires reading the actual image file, which can be slow.
        # A better approach is to have image dimensions pre-computed in the Excel file.
        # TODO: If you have image dimensions in your Excel, modify the code to read from there.
        image_path = os.path.join(IMAGE_DIR, filename)
        try:
            # Using opencv to get image dimensions
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            img_height, img_width, _ = img.shape
        except ImportError:
            print("Warning: OpenCV is not installed. Cannot determine image dimensions.")
            print("Please install it using: pip install opencv-python")
            # As a fallback, you might need to use a fixed size, but this is not recommended.
            # img_width, img_height = 1920, 1080 # Example fixed size
            continue
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            continue

        # Prepare to write to the YOLO label file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_filepath = os.path.join(OUTPUT_DIR, label_filename)

        with open(label_filepath, 'w') as f:
            for _, row in group.iterrows():
                class_name = row[CLASS_NAME_COL]
                bbox_str = row[BBOX_COL]

                # The bbox is stored as a string '[xmin,ymin,w,h]', so we need to parse it.
                # TODO: Adjust parsing if your bbox format is different.
                try:
                    bbox = [float(x) for x in bbox_str.strip('[]').split(',')]
                except (ValueError, AttributeError) as e:
                    print(f"Warning: Could not parse bbox '{bbox_str}' for image {filename}. Error: {e}. Skipping.")
                    continue

                if class_name not in CLASS_MAPPING:
                    print(f"Warning: Unknown class '{class_name}' in image {filename}. Skipping.")
                    continue
                
                class_id = CLASS_MAPPING[class_name]

                # Convert to YOLO format
                yolo_bbox = convert_bbox_to_yolo(img_width, img_height, bbox)

                # Write to file
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

    print("\nProcessing complete.")
    print(f"YOLO format labels have been saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()