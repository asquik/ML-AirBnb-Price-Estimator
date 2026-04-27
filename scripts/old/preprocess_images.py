import os
import glob
from PIL import Image, ImageOps
import multiprocessing
from tqdm import tqdm
import argparse
from pathlib import Path

def process_single_image(args):
    """
    Resizes an image to square using Center Crop to preserve aspect ratio.
    """
    img_path, output_path, size = args
    try:
        with Image.open(img_path) as img:
            # Convert to RGB (handles PNG/RGBA alpha channels)
            img = img.convert("RGB")
            
            # ImageOps.fit crops the image to the target size while preserving aspect ratio.
            # It centers the crop by default (centering=(0.5, 0.5)).
            img_resized = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with high quality
            img_resized.save(output_path, "JPEG", quality=90, optimize=True)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Parallel Image Resizer using Center Crop")
    parser.add_argument("--input_dir", type=str, default="images/all", help="Directory with listing folders")
    parser.add_argument("--output_dir", type=str, default="images/processed_224", help="Flat output directory")
    parser.add_argument("--size", type=int, default=224, help="Target square size")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of CPU workers")
    args = parser.parse_args()

    # Find all images with common extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    
    if not image_paths:
        print(f"No images found in {args.input_dir}.")
        return

    # To ensure one image per listing, group by listing_id (folder name)
    listing_groups = {}
    for img_path in image_paths:
        listing_id = os.path.basename(os.path.dirname(img_path))
        if listing_id not in listing_groups:
            listing_groups[listing_id] = []
        listing_groups[listing_id].append(img_path)

    print(f"Found {len(image_paths)} images across {len(listing_groups)} listings.")
    print(f"Processing one representative image per listing using {args.workers} cores...")

    # Prepare task arguments: Pick the first image alphabetically per listing
    tasks = []
    for listing_id, imgs in listing_groups.items():
        imgs.sort() # Ensure deterministic selection (e.g., 000_... first)
        selected_img = imgs[0]
        # Standardize output name to {listing_id}.jpg for the feature extractor
        output_path = os.path.join(args.output_dir, f"{listing_id}.jpg")
        tasks.append((selected_img, output_path, args.size))

    # Run parallel pool
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks)))

    success_count = sum(results)
    print(f"\nProcessing Complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(tasks) - success_count}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
