import pyscreenshot as pss
import time
from image_helper import grayscale_resized_image
from board_detector import detect_chessboard, get_chess_tiles

while True:
    # grab fullscreen
    img = pss.grab()
    img = grayscale_resized_image(img)

    is_match, lines_x, lines_y = detect_chessboard(img)
    if is_match:
        print("found Chessboard!!")
    else:
        print("Chessboard not detected on screen!")
    time.sleep(0.1)

