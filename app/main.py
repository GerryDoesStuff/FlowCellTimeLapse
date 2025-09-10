from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from .ui.main_window import MainWindow

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.debug("Application initialized")

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
