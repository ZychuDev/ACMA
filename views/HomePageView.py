from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QTableWidget, QTableWidgetItem, QSizePolicy, QFrame, QAbstractItemView
from PyQt6.QtGui import QPixmap, QPaintEvent, QPainter, QFont, QBrush, QColor, QPalette
from PyQt6.QtCore import Qt

from controllers import HomePageController

class ResizableLabel(QLabel):
    def __init__(self, parent:QWidget, pixmap: QPixmap):
        super().__init__(parent)
        self._pixmap = pixmap

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self._pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
class HomePageUi(QWidget):
    def __init__(self):
        super().__init__()
        self.vertical_layout = QVBoxLayout()
        
        title_label = ResizableLabel(self, QPixmap("assets/img/relacs.jpg"))
        title_label.setScaledContents(True)
        title_label.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))


        p: QPalette = self.palette()
        p.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(p)
        self.setAutoFillBackground(True)

        self.lower_horizontal_layout = QHBoxLayout()

        information_table = QTableWidget(11,2)
        information_table.setMinimumWidth(600)
        information_table.setObjectName("InformationTable")
        information_table.setShowGrid(False)
        information_table.setWordWrap(False)
        font = QFont()
        font.setPointSize(20)
        information_table.setFont(font)
        information_table.setFrameShape(QFrame.Shape.NoFrame)
        information_table.setFrameShadow(QFrame.Shadow.Sunken)
        information_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        information_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        information_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        information_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        information_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        information_table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        information_table.setTextElideMode(Qt.TextElideMode.ElideNone)
        headers = information_table.horizontalHeader()
        headers.setVisible(False)
        headers.setCascadingSectionResizes(True)
        headers.setDefaultSectionSize(200)
        headers.setMinimumSectionSize(200)
        headers.setStretchLastSection(True)
        headers = information_table.verticalHeader()
        headers.setVisible(False)
        headers.setCascadingSectionResizes(True)
        headers.setDefaultSectionSize(40)
        headers.setMinimumSectionSize(40)
        headers.setStretchLastSection(True)
        information_table.setSortingEnabled(False)


        information_table.setItem(0, 0, QTableWidgetItem("Name:"))
        information_table.setItem(1, 0, QTableWidgetItem("Build date:"))
        information_table.setItem(2, 0, QTableWidgetItem("Version:"))
        information_table.setItem(3, 0, QTableWidgetItem("License:"))
        information_table.setItem(4, 0, QTableWidgetItem("Reference:"))
        information_table.setItem(5, 0, QTableWidgetItem("DOI:"))
        information_table.setItem(6, 0, QTableWidgetItem("Created by:"))
        information_table.setItem(7, 0, QTableWidgetItem("Logo by:"))
        information_table.setItem(8, 0, QTableWidgetItem("Group name:"))
        information_table.setItem(9, 0,QTableWidgetItem("Website:"))
        information_table.setItem(10, 0,QTableWidgetItem("Contact:"))

        item = QTableWidgetItem("relACs")
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        information_table.setItem(0, 1, item)
        information_table.setItem(1, 1, QTableWidgetItem("12.05.2024")) #VERSION
        information_table.setItem(2, 1, QTableWidgetItem("2.4.9")) #VERSION
        information_table.setItem(3, 1, QTableWidgetItem("GPLv3"))
        information_table.setItem(4, 1, QTableWidgetItem("Chem. Commun., 2022, 58, 6381-6384"))
        item = QTableWidgetItem("10.1039/D2CC02238A")
        item.setForeground(QBrush(QColor(0,0,255)))
        information_table.setItem(5, 1, item)
        information_table.setItem(6, 1, QTableWidgetItem("Wiktor & Mikołaj Żychowicz"))
        information_table.setItem(7, 1, QTableWidgetItem("Robert Jankowski"))
        information_table.setItem(8, 1, QTableWidgetItem("Multifunctional Luminescent Materials Group"))
        item = QTableWidgetItem("www.multilumimater.pl")
        item.setForeground(QBrush(QColor(0,0,255)))
        information_table.setItem(9, 1, item)
        information_table.setItem(10, 1, QTableWidgetItem("mikolaj.zychowicz@uj.edu.pl"))
        
        group_logo_label = ResizableLabel(self, QPixmap("assets/img/logo-2.jpg"))
        group_logo_label.setScaledContents(True)

        self.lower_horizontal_layout.addWidget(information_table, stretch=1)
        self.lower_horizontal_layout.addWidget(group_logo_label, stretch=1)
        self.vertical_layout.addWidget(title_label, stretch=1)
        self.vertical_layout.addLayout(self.lower_horizontal_layout, stretch=1)


    def get_layout(self) -> QVBoxLayout:
        return self.vertical_layout

class HomePageView(QWidget):
    def __init__(self):
        super().__init__()
        self._ui = HomePageUi()

        self.setLayout(self._ui.get_layout())
        self._ui.lower_horizontal_layout.itemAt(0).widget().itemClicked.connect(HomePageController.open_url)
        