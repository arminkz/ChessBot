import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPen, QRegion
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QPushButton
from PyQt5.QtWidgets import QWidget


class ScreenHighlighter(QWidget):
    setVisibleSignal = QtCore.pyqtSignal(bool)
    resizeSignal = QtCore.pyqtSignal(int, int)
    moveSignal = QtCore.pyqtSignal(int, int)
    status_text = ""
    color = QtCore.Qt.red

    x0 = 0
    y0 = 0
    lines_x = []
    lines_y = []

    def safe_setVisible(self, v):
        self.setVisibleSignal.emit(v)

    def safe_resize(self, w, h):
        self.resizeSignal.emit(w, h)

    def safe_move(self, x, y):
        self.moveSignal.emit(x, y)

    def set_status(self, str):
        self.status_text = str

    def set_color(self, color):
        self.color = color

    def set_lines(self, x0, y0, lines_x, lines_y):
        self.x0 = x0
        self.y0 = y0
        self.lines_x = lines_x
        self.lines_y = lines_y

    def __init__(self, parent=None):
        super(ScreenHighlighter, self) \
            .__init__(parent, QtCore.Qt.FramelessWindowHint
                      | QtCore.Qt.WindowSystemMenuHint
                      | QtCore.Qt.WindowStaysOnTopHint
                      | QtCore.Qt.WindowTransparentForInput)

        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.resize(500, 500)

        self.resizeSignal.connect(self.resize)
        self.moveSignal.connect(self.move)
        self.setVisibleSignal.connect(self.setVisible)

        sizeObject = QDesktopWidget().screenGeometry(-1)
        print("Screen size : " + str(sizeObject.width()) + "x" + str(sizeObject.height()))

        self.setWindowTitle(self.tr("ChessBot v0.1"))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(QRect(0, 0, self.geometry().width(), 20),QtCore.Qt.black)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QPen()
        pen.setColor(self.color)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(QRect(0, 0, self.geometry().width(), self.geometry().height()))

        # if len(self.lines_x) > 0:
        #     for i in range(len(self.lines_x)):
        #         painter.drawLine(self.lines_x[i] - self.x0, 20, self.lines_x[i] - self.x0, self.geometry().height())
        #         painter.drawLine(0, self.lines_y[i] - self.y0 + 20, self.geometry().width(), self.lines_y[i] - self.y0 + 20)

        if self.status_text != "":
            painter.drawText(5, 15, self.status_text)

    def sizeHint(self):
        return QtCore.QSize(200, 200)

    def resizeEvent(self, event):
        pass
        # r = 3
        # w = self.frameGeometry().width()
        # h = self.frameGeometry().height()
        # reg = QRegion(self.frameGeometry()) \
        #     .subtracted(
        #     QRegion(r, r, w - 2 * r, h - 2 * r, QtGui.QRegion.Rectangle))
        # self.setMask(reg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    sh = ScreenHighlighter()
    sh.show()
    sys.exit(app.exec_())
