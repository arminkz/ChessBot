import numpy as np
import PIL
from tensorflow.keras.models import load_model
from image_helper import grayscale_resized_image
from board_detector import detect_chessboard, get_chess_tiles


# load pre-trained model
model = load_model('models/softmax_v1')
model.summary()

label = ' KQRBNPkqrbnp'


# reads tiles object and returns FEN string
def predict(tiles):
    fen = ''
    c = 0
    print(tiles.shape)
    for i in range(8):
        if i != 0 and i != 7:
            fen += '/'
        for j in range(8):
            img_arr = tiles[:, :, i * 8 + j]
            img = PIL.Image.fromarray(img_arr).resize([32, 32], PIL.Image.ADAPTIVE)
            img_arr = grayscale_resized_image(img)[:, :] / 255.0
            img_arr = img_arr.reshape(1, 32, 32)
            cls = np.argmax(model.predict(img_arr), axis=-1)[0]
            if cls == 0:
                c += 1
            else:
                if c != 0:
                    fen += str(c)
                    c = 0
                fen += label[cls]
        if c != 0:
            fen += str(c)
            c = 0
    return fen


def test_predictor():
    image_from_file = PIL.Image.open("example_screens/lichess_screen.png")
    img = grayscale_resized_image(image_from_file)
    is_match, lines_x, lines_y = detect_chessboard(img)
    if is_match:
        tiles = get_chess_tiles(img, lines_x, lines_y)
        print("found Chessboard!!")
        print(predict(tiles))
    else:
        print("Chessboard not detected on screen!")


if __name__ == "__main__":
    test_predictor()
