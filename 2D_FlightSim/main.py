"""
main.py — Entry point for the 2D Flight Simulator

Run:
    python main.py
"""

import sys
from PyQt5.QtWidgets import QApplication
from Gui import FlightSimWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("2D Flight Simulator")
    window = FlightSimWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()