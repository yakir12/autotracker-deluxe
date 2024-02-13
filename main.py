# libraries
import os
import tkinter as tk
from tkinter import filedialog
import shutil

from calibration import make_checkerboard, calib
from autotrack import autotracker
#from analysis import analyze

import cv2
import numpy as np


# Get Username and Session ID, create folders
input_str1 = ''
input_str2 = ''

def get_input():
    global input_str1
    global input_str2
    input_str1 = input_entry1.get()
    input_str2 = input_entry2.get()
    print(f"Username entered: {input_str1} \nSession ID entered: {input_str2} \n\nUser information obtained, you may now close the dialog box to proceed")

# Create a new Tkinter window
root = tk.Tk()
root.title('User information')
root.geometry('300x200+500+200')

# Create label widgets to display some text
label1 = tk.Label(root, text="Enter username:")
label2 = tk.Label(root, text="Enter session ID:")

# Create entry widgets to accept user input
input_entry1 = tk.Entry(root)
input_entry2 = tk.Entry(root)

# Create a button widget to submit the user input
submit_button = tk.Button(root, text="Submit", command=get_input)

# Add the widgets to the window
label1.pack()
input_entry1.pack()
label2.pack()
input_entry2.pack()
submit_button.pack()

# Start the main event loop
root.mainloop()

dir = 'data/' + input_str1 + '/' + input_str2 + '/'

dir_calib = dir + 'calib_data/'
dir_tracking = dir + 'tracking_data/'
dir_analysis = dir + 'analysis_data/'


if not os.path.exists('data/' + input_str1 + '/'):
    os.makedirs('data/' + input_str1 + '/')
else:
    pass

if not os.path.exists(dir):
    os.mkdir(dir)
else:
    pass

if not os.path.exists(dir_calib):
    os.mkdir(dir_calib)
else:
    pass

if not os.path.exists(dir_tracking):
    os.mkdir(dir_tracking)
else:
    pass

if not os.path.exists(dir_analysis):
    os.mkdir(dir_analysis)
else:
    pass



# Calibration video upload
class FileCopyGUI:
    def __init__(self, master):
        self.master = master
        master.title("Calibration video upload")

        # Set up the GUI widgets
        self.source_label = tk.Label(master, text="Source directory:")
        self.source_label.grid(row=0, column=0)

        self.source_entry = tk.Entry(master)
        self.source_entry.grid(row=0, column=1)

        self.source_button = tk.Button(master, text="Browse", command=self.browse_source)
        self.source_button.grid(row=0, column=2)

        self.copy_button = tk.Button(master, text="Upload file", command=self.copy_file)
        self.copy_button.grid(row=2, column=1)

        self.quit_button = tk.Button(master, text="Skip", command=self.master.destroy)
        self.quit_button.grid(row=3, column=1)

    def browse_source(self):
        source_path = filedialog.askopenfilename()
        self.source_entry.delete(0, tk.END)
        self.source_entry.insert(0, source_path)

    def copy_file(self):
        global format_calibration
        source_path = self.source_entry.get()
        dest_path = dir
        shutil.copy(source_path, dest_path)
        source_filename, source_ext = os.path.splitext(source_path)
        format_calibration = source_ext[1:]
        with open(dir + 'calib_data/' + 'format_calibration' + '.txt', "w") as text_file:
            text_file.write(format_calibration)
        print("\nCalibration video uploaded")

root = tk.Tk()
root.geometry('400x100+500+200')
app = FileCopyGUI(root)
root.mainloop()


# Raw video upload
class FileCopyGUI2:
    def __init__(self, master):
        self.master = master
        master.title("Raw video upload")

        # Set up the GUI widgets
        self.source_label = tk.Label(master, text="Source directory:")
        self.source_label.grid(row=0, column=0)

        self.source_entry = tk.Entry(master)
        self.source_entry.grid(row=0, column=1)

        self.source_button = tk.Button(master, text="Browse", command=self.browse_source)
        self.source_button.grid(row=0, column=2)

        self.copy_button = tk.Button(master, text="Upload file", command=self.copy_file)
        self.copy_button.grid(row=2, column=1)

        self.quit_button = tk.Button(master, text="Skip", command=self.master.destroy)
        self.quit_button.grid(row=3, column=1)

    def browse_source(self):
        source_path = filedialog.askopenfilename()
        self.source_entry.delete(0, tk.END)
        self.source_entry.insert(0, source_path)

    def copy_file(self):
        global format_calibration
        source_path = self.source_entry.get()
        filetype = source_path.split(".")[-1]
        filename = "raw." + filetype
        dest_path = dir + filename
        shutil.copy(source_path, dest_path)
        source_filename, source_ext = os.path.splitext(source_path)
        format_raw = source_ext[1:]
        with open(dir + 'calib_data/' + 'format_raw' + '.txt', "w") as text_file:
            text_file.write(format_raw)
        print("\nRaw video uploaded")

root = tk.Tk()
root.geometry('400x100+500+200')
app = FileCopyGUI2(root)
root.mainloop()



# Calibration, Tracking and Analysis
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.function_label = tk.Label(self, text="a) You are running main.py in the directory Dropbox/Tracking/offline \nb) Run the functions below in order, as the results of a step are used as inputs to successive steps. \nc) The directory structure for data storage and retrieval is [current directory]/data/[user]/[session]/ (if you are looking to access the files) \nd) The results of each step are automatically saved to the above folder when completed, so you can quit and resume from that step at a later time without loss of data. \n\nSelect a function and click Run to proceed:", justify='left')
        self.function_label.pack()

        self.function_var = tk.StringVar(value="0. Define parameters")
        self.function0 = tk.Radiobutton(self, text="0. Choose checkerboard type", variable=self.function_var, value="0. Define parameters")
        self.function0.pack(anchor="w")
        self.function1 = tk.Radiobutton(self, text="1. Extract calibration parameters", variable=self.function_var, value="1. Extract calibration parameters")
        self.function1.pack(anchor="w")
        self.function2 = tk.Radiobutton(self, text="2. Calibrate video", variable=self.function_var, value="2. Calibrate video")
        self.function2.pack(anchor="w")
        self.function3 = tk.Radiobutton(self, text="3. Tracking", variable=self.function_var, value="3. Tracking")
        self.function3.pack(anchor="w")
        self.function4 = tk.Radiobutton(self, text="4. Analysis", variable=self.function_var, value="4. Analysis")
        self.function4.pack(anchor="w")

        self.run_button = tk.Button(self, text="Run", command=self.run_function)
        self.run_button.pack()

        self.quit_button = tk.Button(self, text="Quit", command=self.master.destroy)
        self.quit_button.pack()

    def run_function(self):
        selected_function = self.function_var.get()
        if selected_function == "0. Define parameters":
            self.function0_call()
        elif selected_function == "1. Extract calibration parameters":
            self.function1_call()
        elif selected_function == "2. Calibrate video":
            self.function2_call()
        elif selected_function == "3. Tracking":
            self.function3_call()
        elif selected_function == "4. Analysis":
            self.function4_call()

    def function0_call(self):
        checkerboard_window = tk.Toplevel()
        checkerboard_window.geometry('400x150+500+200')
        def_params(master=checkerboard_window)


    def function1_call(self):

        with open(dir + 'calib_data/' + 'format_calibration' + '.txt', "r") as text_file:
            format_calibration = text_file.read()

        n_rows = np.load(dir + 'calib_data/' + 'n_rows.dat', allow_pickle=True)
        n_columns = np.load(dir + 'calib_data/' + 'n_columns.dat', allow_pickle=True)
        square_size = np.load(dir + 'calib_data/' + 'square_size.dat', allow_pickle=True)

        checkerboard, checkerboard_size = make_checkerboard(n_rows, n_columns, square_size)
        calib(dir, format_calibration, checkerboard, checkerboard_size)

        print("\nCalibration parameters extracted!")

    def function2_call(self):

        with open(dir + 'calib_data/' + 'format_raw' + '.txt', "r") as text_file:
            format_raw = text_file.read()

        mapx = np.load(dir + 'calib_data/' + 'mapx.dat', allow_pickle=True)
        mapy = np.load(dir + 'calib_data/' + 'mapy.dat', allow_pickle=True)
        frame_size = np.load(dir + 'calib_data/' + 'frame_size.dat', allow_pickle=True)
        H = np.load(dir + 'calib_data/' + 'H.dat', allow_pickle=True)

        cap = cv2.VideoCapture(dir + 'raw' + '.' + format_raw)
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps_processed = fps_raw

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result = cv2.VideoWriter(dir + 'processed' + '.' + 'mp4', fourcc, fps_processed, frame_size)

        cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame_dst = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

                frame_height,  frame_width = frame_dst.shape[:2]
                frame_warp = cv2.warpPerspective(frame_dst, H, (frame_width, frame_height))

                cv2.imshow('Processed',frame_warp)
                result.write(frame_warp)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break

        cap.release()
        result.release()
        cv2.destroyAllWindows()

    def function3_call(self):
        print("3. Tracking called!")

        desired_tracker = 'BOOSTING'
        track_filename = 'processed'
        format_track = 'mp4'
        autotracker(dir, track_filename, input_str2, format_track, desired_tracker)


    def function4_call(self):

        with open(dir + 'tracking_data/' + 'coordinates' + '.csv', newline='') as csvfile:
            r = np.loadtxt(csvfile,delimiter=',')

        with open(dir + 'tracking_data/' + 'time' + '.csv', newline='') as csvfile:
            tau = np.loadtxt(csvfile,delimiter=',')

        with open(dir + 'tracking_data/' + 'track_split_time' + '.csv', newline='') as csvfile:
            tau_split = np.loadtxt(csvfile,delimiter=',')

        with open(dir + 'tracking_data/' + 'fps' + '.csv', newline='') as csvfile:
            fps = np.loadtxt(csvfile,delimiter=',')

        analyze(dir, r, tau, tau_split, fps)


# Checkerboard selection widget
class def_params(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.function_label = tk.Label(self, text="Select a checkerboard:")
        self.function_label.pack()

        self.function_var = tk.StringVar(value="Checkerboard 3")
        self.function0 = tk.Radiobutton(self, text="Checkerboard 1 (7 by 9)", variable=self.function_var, value="Checkerboard 1")
        self.function0.pack(anchor="w")
        self.function1 = tk.Radiobutton(self, text="Checkerboard 2 (13 by 23)", variable=self.function_var, value="Checkerboard 2")
        self.function1.pack(anchor="w")
        self.function1 = tk.Radiobutton(self, text="Checkerboard 3 (6 by 9)", variable=self.function_var, value="Checkerboard 3")
        self.function1.pack(anchor="w")

        self.function2 = tk.Label(self, text="Enter checkerboard size (mm):")
        self.function2.pack(anchor="w")
        self.checkerboard_size_entry = tk.Entry(self)
        self.checkerboard_size_entry.pack(anchor="w")


        self.run_button = tk.Button(self, text="Make selection", command=self.run_function)
        self.run_button.pack()


    def run_function(self):
        selected_function = self.function_var.get()
        square_size = self.checkerboard_size_entry.get()
        square_size = int(square_size)

        if selected_function == "Checkerboard 1":
            n_rows = 7
            n_columns = 9
            # square_size = checkerboard_size_entry
            print("\nCheckerboard 1 selected!\nCheckerboard size:",square_size,'mm')

        elif selected_function == "Checkerboard 2":
            n_rows = 13
            n_columns = 23
            # square_size = checkerboard_size_entry
            print("\nCheckerboard 2 selected!\nCheckerboard size:",square_size,'mm')

        elif selected_function == "Checkerboard 3":
            n_rows = 6
            n_columns = 9
            # square_size = checkerboard_size_entry
            print("\nCheckerboard 3 selected!\nCheckerboard size:",square_size,'mm')

        n_rows = np.array(n_rows)
        n_columns = np.array(n_columns)
        square_size = np.array(square_size)

        n_rows.dump(dir + 'calib_data/' + 'n_rows.dat')
        n_columns.dump(dir + 'calib_data/' + 'n_columns.dat')
        square_size.dump(dir + 'calib_data/' + 'square_size.dat')

        self.master.destroy()



root = tk.Tk()
root.title('Offline tracking')
app = Application(master=root)
app.mainloop()
