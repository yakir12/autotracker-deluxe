import tkinter as tk
import autocalibration
import calibration as calib

class AutocalibrationTool(tk.TopLevel):
    def __init__(self, parent):
        super().__init__(self, parent)
        self.title("Autocalibration")

    
