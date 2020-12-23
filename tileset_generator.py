import os
import glob
import numpy as np
import PIL
from image_helper import load_image_grayscale
from board_detector import detect_chessboard, get_chess_tiles


def save_tiles(tiles, img_save_dir, img_file):
    letters = 'ABCDEFGH'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    for i in range(64):
        sqr_filename = f"{img_save_dir}/{img_file}_{letters[i % 8]}{int(i / 8) + 1}.png"

        # Make resized 32x32 image from matrix and save
        if tiles.shape != (32, 32, 64):
            PIL.Image.fromarray(tiles[:, :, i]) \
                .resize([32, 32], PIL.Image.ADAPTIVE) \
                .save(sqr_filename)
        else:
            # Possibly saving floats 0-1 needs to change fromarray settings
            PIL.Image.fromarray((tiles[:, :, i] * 255).astype(np.uint8)) \
                .save(sqr_filename)


def generate_tileset(input_chessboard_folder, output_tile_folder):
    # Create output folder as needed
    if not os.path.exists(output_tile_folder):
        os.mkdir(output_tile_folder)

    # Get all image files of type png/jpg/gif
    img_files = set(glob.glob("%s/*.png" % input_chessboard_folder))\
        .union(set(glob.glob("%s/*.jpg" % input_chessboard_folder)))\
        .union(set(glob.glob("%s/*.gif" % input_chessboard_folder)))

    num_success = 0
    num_failed = 0
    num_skipped = 0

    for i, img_path in enumerate(img_files):
        print("------")
        print(f"{i+1}/{len(img_files)} : {img_path}")
        # Strip to just filename
        img_file = img_path[len(input_chessboard_folder)+1:-4]

        # Create output save directory or skip this image if it exists
        img_save_dir = f"{output_tile_folder}/tiles_{img_file}"

        if os.path.exists(img_save_dir):
            print("\tSkipping existing")
            num_skipped += 1
            continue

        # Load image
        print(f"Loading {img_path}...")
        img_arr = np.array(load_image_grayscale(img_path), dtype=np.float32)

        # Get tiles
        print(f"Generating tiles for {img_file}...")
        is_match, lines_x, lines_y = detect_chessboard(img_arr)

        if is_match:
            tiles = get_chess_tiles(img_arr, lines_x, lines_y)
            print(f"Saving tiles {img_file}...")
            save_tiles(tiles, img_save_dir, img_file)
            num_success += 1
        else:
            print("No Match, skipping")
            num_failed += 1

    print(f"\t{num_success}/{len(img_files) - num_skipped} generated, {num_failed} failures, {num_skipped} skipped.")