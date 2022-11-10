from PyQt6.QtGui import QColor, QBrush
from .StandardItem import StandardItem
from .CompundItemsCollection import CompoundItemsCollection
from models import CompoundsCollection
from controllers import CompoundItemsCollectionController

class RootItem(StandardItem):
    def __init__(self, txt:str='', font_size:int=16, set_bold:bool=True, color:QColor=QColor(0,0,0)):
        super().__init__(txt, font_size, set_bold, color)
        self.setBackground(QBrush(QColor(255,122,0)))

        model = CompoundsCollection("AAA")
        self.compounds = CompoundItemsCollection(model, CompoundItemsCollectionController(model))
        
        self.appendRow(self.compounds)
