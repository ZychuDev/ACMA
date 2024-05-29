"""
    The relACs is a analysis tool for magnetic data for SMM systems using
    various models for ac magnetic characteristics and the further reliable
    determination of diverse relaxation processes.

    Copyright (C) 2023  Wiktor Zychowicz & Mikolaj Zychowicz

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
""" 

import sys

import requests #type: ignore
from webbrowser import open

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QIcon

from views import MainView


class RelACs(QApplication):
    """Implementation of entire program.

    """
    def __init__(self, sys_argv:list[str]):
        super(RelACs, self).__init__(sys_argv)
        self.main_view: MainView = MainView()
        app_icon: QIcon = QIcon("./assets/img/relACs.ico")
        RelACs.setWindowIcon(app_icon) 
        self.main_view.showMaximized()

class Launcher(QApplication):
    """Checks if installed relACs version is actual. Helps install the latest version.
    """
    def __init__(self, sys_argv:list[str]):
        super(Launcher, self).__init__(sys_argv)
        self.window = QWidget()
        app_icon: QIcon = QIcon("./assets/img/relACs.ico")
        Launcher.setWindowIcon(app_icon)
        launcher_layout = QVBoxLayout()
        launcher_layout.addWidget(QLabel(f"New relACs version {latest_relase} is available."))
        install_button = QPushButton("Install new version")
        install_button.clicked.connect(lambda: open(f"https://github.com/ZychuDev/relACs/releases/tag/{latest_relase}")) #type: ignore
        skip_button: QPushButton = QPushButton("Continue using old version")
        skip_button.clicked.connect(Launcher.closeAllWindows) #type: ignore
        launcher_layout.addWidget(install_button)
        launcher_layout.addWidget(skip_button)
        self.window.setLayout(launcher_layout)
        self.window.show()

if __name__ == "__main__":
    response = requests.get("https://api.github.com/repos/ZychuDev/relACs/releases")
    if response.ok:
        release = "2.4.9" #VERSION
        latest_relase= response.json()[0]["tag_name"]
        if latest_relase != release:
            launcher = Launcher(sys.argv)
            launcher.exec()
    app:QApplication = RelACs(sys.argv)
    # app.setStyle("Fusion")
    sys.exit(app.exec())