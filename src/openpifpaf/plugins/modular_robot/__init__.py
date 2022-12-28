import openpifpaf

from . import modular_robotkp


def register():
    openpifpaf.DATAMODULES['modular_robot'] = modular_robotkp.ModularRobotKp
