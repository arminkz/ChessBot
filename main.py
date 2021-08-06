import subprocess
import sys
import time
import PIL
import numpy as np
import pyautogui
import pyscreenshot as pss
import matplotlib.pyplot as plt
import tensorflow as tf
import chess
import chess.engine

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

from board_detector import detect_chessboard, get_chess_tiles
from image_helper import grayscale_image
from screen_helper import ScreenHighlighter

app = QApplication(sys.argv)
screen_h = ScreenHighlighter()
screen_h.show()
screen_h.safe_resize(300, 20)
screen_h.safe_move(0, 0)
screen_h.safe_setVisible(False)

rect_x0 = None
rect_y0 = None
rect_w = None
rect_h = None

best_move = None

ch_lines_x = None
ch_lines_y = None

te_log = None

engine = None
model = tf.keras.models.load_model('models/softmax_v1')


def window():
    global te_log, engine
    widget = QWidget()
    widget.setFixedSize(310, 200)

    b_locate = QPushButton('', widget)
    b_locate.move(0, 0)
    b_locate.clicked.connect(b_locate_clicked)
    b_locate.setIcon(QIcon('ui/board.png'))
    b_locate.setIconSize(QSize(20, 20))

    b_test1 = QPushButton('', widget)
    b_test1.move(60, 0)
    b_test1.clicked.connect(test1)
    b_test1.setIcon(QIcon('ui/board-full.png'))
    b_test1.setIconSize(QSize(20, 20))

    b_play = QPushButton('', widget)
    b_play.move(120,0)
    b_play.clicked.connect(b_auto_play_clicked)
    b_play.setIcon(QIcon('ui/play.png'))
    b_play.setIconSize(QSize(20, 20))

    b_exit = QPushButton('', widget)
    b_exit.move(180, 0)
    b_exit.clicked.connect(b_exit_clicked)
    b_exit.setIcon(QIcon('ui/exit.png'))
    b_exit.setIconSize(QSize(20, 20))

    te_log = QTextEdit('ChessBot v1.0 by @arminkz', widget)
    te_log.move(5, 38)
    te_log.setReadOnly(True)
    te_log.setFixedSize(300, 150)

    widget.setGeometry(50, 50, 320, 200)
    widget.setWindowTitle("Chessbot")
    widget.show()

    logln("")
    logln("[Engine] Starting Stockfish engine ...")
    engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
    sys.exit(app.exec_())


def b_auto_play_clicked():
    print("TODO")


def b_locate_clicked():
    global rect_x0, rect_y0, rect_w, rect_h
    global ch_lines_x, ch_lines_y
    logln("[Vision] Detecting chessboard on screen ...")
    img = pss.grab()
    img = img.resize([int(0.5 * s) for s in img.size])
    img = grayscale_image(img)

    is_match, lines_x, lines_y = detect_chessboard(img)
    if is_match:
        print(lines_x)
        print(lines_y)

        ch_lines_x = lines_x
        ch_lines_y = lines_y

        # Highlight Detected Area
        screen_h.safe_setVisible(True)

        stepx = np.int32(np.round(np.mean(np.diff(lines_x))))
        stepy = np.int32(np.round(np.mean(np.diff(lines_y))))

        x0 = lines_x[0] - stepx
        y0 = lines_y[0] - stepy
        w = lines_x[-1] - x0 + stepx
        h = lines_y[-1] - y0 + stepy

        rect_x0 = x0
        rect_y0 = y0
        rect_w = w
        rect_h = h

        screen_h.set_lines(x0, y0, lines_x, lines_y)
        screen_h.safe_move(x0 - 1, y0 - 21)
        screen_h.safe_resize(w + 2, h + 22)

        tiles = get_chess_tiles(img, lines_x, lines_y)
        logln(f"[Vision] Chessboard has been detected!")
        screen_h.set_status(f"Chessboard detected")
        screen_h.set_color(QtCore.Qt.green)
        screen_h.safe_setVisible(True)
        screen_h.update()
    else:
        logln(f"[Vision] Error detecting chessboard!")
        # screen_h.set_color(QtCore.Qt.red)
        # screen_h.safe_move(0, 20)
        # screen_h.safe_resize(300, 20)
        screen_h.safe_setVisible(False)
        # screen_h.update()


def b_exit_clicked():
    app.exit(1)


def logln(str):
    global te_log
    te_log.setText(te_log.toPlainText() + "\n" + str)
    te_log.verticalScrollBar().setValue(te_log.verticalScrollBar().maximum())


def log(str):
    global te_log
    te_log.setText(te_log.toPlainText() + str)
    te_log.verticalScrollBar().setValue(te_log.verticalScrollBar().maximum())


def display_array(a, rng=[0, 1]):
    a = (a - rng[0]) / float(rng[1] - rng[0]) * 255
    a = np.uint8(np.clip(a, 0, 255))
    plt.imshow(PIL.Image.fromarray(a))
    plt.show()


def test1():
    global best_move
    # print('model loaded')
    # print(model.summary())
    logln("[Piece Detection] Predicting using neural net ...")
    img = pss.grab()
    img = img.resize([int(0.5 * s) for s in img.size])
    img = grayscale_image(img)
    tiles = get_chess_tiles(img, ch_lines_x, ch_lines_y)
    # print(tiles.shape)
    norm_tiles = np.zeros((64, 32, 32))
    for i in range(64):
        resized_image = PIL.Image.fromarray(tiles[:, :, i]).resize([32, 32])
        norm_tiles[i, :, :] = np.array(resized_image) / 255.0
    # print(norm_tiles.shape)
    preds = model.predict(norm_tiles)
    labels = np.argmax(preds, axis=1)

    logln("[Piece Detection] FEN: ")
    fen_text = label2FEN(labels)
    log(fen_text)

    # feed FEN to stockfish
    board = chess.Board(f"{fen_text} w")
    result = engine.play(board, chess.engine.Limit(time=0.1))
    logln(f"[Engine] Best Move: {result.move}")
    best_move = result.move

    move = str(best_move)
    start_coord = move[0:2]
    end_coord = move[2:4]
    sx = ord(start_coord[0]) - ord('a')
    sy = 8 - int(start_coord[1])
    ex = ord(end_coord[0]) - ord('a')
    ey = 8 - int(end_coord[1])
    cell = rect_w / 8
    tsx = rect_x0 + sx*cell + (cell/2)
    tsy = rect_y0 + sy*cell + (cell/2)
    tex = rect_x0 + ex*cell + (cell/2)
    tey = rect_y0 + ey*cell + (cell/2)
    pyautogui.click(tsx, tsy, clicks=2, interval=0.2)
    time.sleep(0.3)
    pyautogui.dragTo(tex, tey, button='left', duration=0.3)
    time.sleep(1.5)
    test1()



def label2FEN(labels):
    fen = ""
    spc = 0
    lastsp = False
    for i in range(8):
        for j in range(8):
            lbl = ' KQRBNPkqrbnp'[labels[(7 - i) * 8 + j]]
            if lbl == ' ':
                lastsp = True
                spc += 1
            else:
                if lastsp:
                    fen += str(spc)
                    spc = 0
                    lastsp = False
                fen += lbl
        if lastsp:
            fen += str(spc)
            spc = 0
            lastsp = False
        if i != 7: fen += '/'
    return fen


if __name__ == '__main__':
    window()
