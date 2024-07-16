"""
window_tester.py

Test file which can be very useful for testing new windows without having to
interact with the rest of the UI. (Can speed up testing greatly.)
"""

from autocalibration import check_calibration
from dtrack_params import dtrack_params
import calibration as calib
import os

from configuration_tool import ConfigurationTool
import tkinter as tk

window = ConfigurationTool(tk.Tk())
window.mainloop()