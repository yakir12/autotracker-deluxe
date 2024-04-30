from autocalibration import check_calibration
from dtrack_params import dtrack_params
import calibration as calib
import os


ext_image_path = os.path.join(
            dtrack_params["project_directory"],
            "calibration_cache",
            "extrinsic",
            "000.png")
calib_filepath = os.path.join(
            dtrack_params["project_directory"],
            "calibration_cache",
            "calibration.dt2c")
check_calibration(ext_image_path, calib.from_file(calib_filepath))