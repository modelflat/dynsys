import pyopencl as cl
from PyQt5.Qt import QApplication, QDesktopWidget
import sys
import argparse

class OpenCLApp(QApplication):

    def __init__(self, title):
        self.app = QApplication(sys.argv)
        # noinspection PyArgumentList
        super().__init__(parent=None)
        self.setWindowTitle(title)

        self.ctx, self.queue = OpenCLApp._createContextAndQueue(
            OpenCLApp._readConfig(
                arg
            )
        )

    @staticmethod
    def _readConfig(self, path):
        import json
        try:
            with open(path) as f: return json.load(f)
        except Exception as e:
            raise RuntimeWarning("Cannot load configuration from file %s: %s" % (path, str(e)))

    @staticmethod
    def _createContextAndQueue(self, config):
        def getAlternatives(d: dict, *alternatives):
            for alt in alternatives:
                val = d.get(alt)
                if val is not None:
                    return val
            raise RuntimeError("No alternative key were found in given dict (alt: {})".format(str(alternatives)))

        if config is None or bool(config.get("autodetect")):
            ctx = cl.create_some_context(interactive=False)
        else:
            pl = cl.get_platforms()[getAlternatives(config, "pid", "platform", "platformId")]
            dev =  pl.get_devices()[getAlternatives(config, "did", "device", "deviceId")]
            ctx = cl.Context([dev])
        return ctx, cl.CommandQueue(ctx)


    def run(self):
        screen = QDesktopWidget().screenGeometry()
        self.show()
        x = ((screen.width() - self.width()) // 2) if screen.width() > self.width() else 0
        y = ((screen.height() - self.height()) // 2) if screen.height() > self.height() else 0
        self.move(x, y)
        sys.exit(self.app.exec_())
