import openpifpaf

from . import roombotkp


def register():
    openpifpaf.DATAMODULES['Roombot'] = roombotkp.RoombotKp

