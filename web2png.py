import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings


class QWebScreenshot(QWebEngineView):

    def capture(self, url, output_file):
        self.output_file = output_file
        self.load(QUrl(url))
        self.loadFinished.connect(self.on_loaded)
        # Create hidden view without scrollbars
        self.setAttribute(Qt.WA_DontShowOnScreen)
        self.page().settings().setAttribute(QWebEngineSettings.ShowScrollBars, False)
        self.show()

    def on_loaded(self):
        size = self.page().contentsSize().toSize()
        self.resize(size)
        # Wait for resize
        QTimer.singleShot(1000, self.take_screenshot)

    def take_screenshot(self):
        self.grab().save(self.output_file, b'PNG')
        self.app.exit(0)


app = QApplication(sys.argv)


def take_chess_screenshot(fen_string=None, output_filename=None):
    # Take uncropped screenshot of lichess board of FEN string and save to file
    url_template = "http://en.lichess.org/editor/%s"
    s = QWebScreenshot()
    s.app = app
    s.capture(url_template % fen_string, output_filename)
    return app.exec_()


def take_screenshot(url=None, output_filename=None):
    s = QWebScreenshot()
    s.app = app
    s.capture(url, output_filename)
    return app.exec_()


# take_chess_screenshot(fen_string='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR',output_filename='webpage.png')
# take_chess_screenshot(fen_string='rnbqkbnr/pppppppp/7Q/8/4P3/8/PPPP1PPP/RNBQKBNR',output_filename='webpage2.png')
