# Autotracker Deluxe

This repo is for the 'Autotracker deluxe' project wherein we are
trying to get a viable autotracking solution up and running for the
dung beetle group. 



## Installation
### Prerequisites

This software depends on Python 3 and pip. You must have these
installed to run the software (and install dependencies). 

If you've installed python and pip and the commands are giving you errors, try 
`python3` and `pip3` instead.

The prefix `$:` indicates a terminal command. You should enter everything after the colon.

### Download

1. Click the 'Code' dropdown above.
2. Click 'Download ZIP'
3. Extract the Downloaded zip file and put the folder containing the code somewhere sensible.

You can also clone the repo using Git if you know how to do this.

### Opening a terminal in the right place

You can navigate to the code directory using the
command line if you know how to do this. Otherwise, you can open
a terminal via the graphical file explorer.

You will need to do this every time you want to use the software.

#### MacOS
Open Finder and find the code folder you just downloaded.
Hold control and click on the folder, then select
'Services' -> 'New Terminal at Folder'.

A terminal window should open.

#### Linux
Open your file explorer, find the code folder you just downloaded.
Right click and select 'Open in Terminal'.

**Note:** This may be distribution dependent.

### Installing dependencies

**Important! :** The tracker depends on the `opencv-contrib-python` package. If you already have a version of opencv installed via pip then you either need to uninstall that version of OpenCV or set up a virtual environment (see below). The `opencv-contrib-python` package should subsume the standard `opencv-python` package so the switch shouldn't cause problems with any other software.

In the terminal, run:

`$: pip install -r requirements.txt`

and press enter. You should only need to do this the first time you download the software.

---
#### (Optional) Using a virtual environment
You may want to use a [virtual environment](https://docs.python.org/3/library/venv.html) for sequestration or testing. This creates a sandboxed python environment where modifications will not interact with your main python installation.

At the terminal (MacOS and Linux), run the following:

`$: mkdir ~/venvs/`

`$: python -m venv ~/venvs/autotracker-deluxe --clear`


This will create a virtual environment named `autotracker-deluxe`.
To activate the environment, you need to use:

`$: source ~/venvs/autotracker-deluxe/bin/activate`

You should see `(autotracker-deluxe)` at your terminal prompt, for example:

`(autotracker-deluxe) user@macbook:~/$ `

Install the dependencies with the environment active. You'll need to activate the environment every time you want to run the software.

You can deactivate the environment using:

`$: deactivate`

You can use [Anaconda](https://www.anaconda.com/download) to achieve the same effect if you know what you're doing.

---

## Usage instructions

Open the code folder in the terminal as you did during the installation process (either using the command line or the graphical file explorer).

### Startup and project selection
In the terminal, run: 
   
   `$: python main.py` 
   
   The main window should open:

![Main window](images/main_window.png)   

Use the 'Select project' button to select an existing project, 'New project' if you wish to create a new project. In either case, a graphical selector will open to allow you to select the project *directory* (the folder in which all project related stuff will be stored).

**Note**: The directory selector is a little clunky, you need to be in the directory you want to select before confirming.

If you are creating a new project, or the project has never been opened with this version of the software, you will be asked if you want to create a project file. Click yes/ok.

Any tools you run will apply only to the currently active project.

Once a project directory is selected, it should be displayed in the project directory entry in the main window.

### Tools
Each stage can be engaged independently by selecting the appropriate radio button and clicking 'Run'. The instructions below assume you have done this.

#### 1. Choose video files
This tool will open a small video selector window.

![Video selection window](images/video_selector.png)

Clicking 'Select' for calibration or tracking will open a file dialog to allow you to select the video. You can also copy the path into the entry but this is not recommended. 

The complete path to the video files is stored, so if the file location is not going to change, you can simply reference the files elsewhere on your hard drive. However, if the file location may change or you are working off of an external hard drive, you may want to make local copies. To do this, check the 'Copy video files locally?' option. This will make copies of the videos in the project directory.

Once you have selected your video files and determined whether you wish to make local copies, click 'Confirm'. Clicking 'Cancel' will discard all changes. 

**Note**: If you copied the files locally then the video filepaths stored by the software will point to the local copies. Thus, if you run the video selection tool again, the entries should show the local copy path rather than the one you originally selected.

#### 2. Configure calibration board
This tool will open a calibration board selector window.

![Calibration selector](images/calibration_window.png)

You can use the spinboxes at the top to select the dimensions of your calibration board. The window will update to show you a visualisation of the pattern. The OpenCV chessboard detection will be overlayed on the image so you can see what pattern the software is actually looking for during calibration.

Changing the square size will change the visible size of the board pattern but pixel size on screen does not matter, this is just for visual feedback. The square size will be used internally to estimate the homography between object coordinates and image coordinates. 

Once you have selected your rows, columns and square size, click 'Confirm'. Clicking 'Cancel' will discard any changes.

#### 3. Extract calibration parameters
1. Follow the on-screen instructions.
2. When the process is finished, a calibrated frame will be shown. Press q to close the window.

**Note:** You should select only a single 'ground' frame. 

#### 4. Autotracker
1. A new window will open playing the video to be tracked (at high speed)
2. Let the video play or skip to a point you wish to start tracking
3. Press P to pause the video, then T to start a track.
4. Select a rectangle (region of interest, ROI) which contains the beetle (be generous). The selection starts from the centre of the region so click and drag from the beetle. Press Enter and close the window. 
5. Press P to play. The bounding box will be overlayed on the video and should track the beetle. The point which is tracked is the centre of the bounding box which is also drawn.
6. Press T to finish the current track.
7. Goto (2)

You can produce any number of tracks from the video. The tracks will be appended
to the track file (`<project_directory>/raw_tracks.csv`). 

**Notes**
1. If you want to cancel your ROI selection, press C then close the window.
2. If you press T by mistake and don't want to track, press T again to cancel. If you didn't play the video while tracking was enabled, the track file won't be modified.
3. You cannot use Q to quit while the video is paused. You can just close the window.
4. Do not use the trackbar while tracking. This will probably break things.
5. The coordinates produced are the centre of mass of the 'blob' which contains
   the beetle and its ball on a given frame. 

#### 5. Process tracks
This option runs in full when you click 'Run'. At present this will:
1. Zero the tracks (translate them so they all start at the same origin, (0,0)).
2. Smooth the tracks using a basic univariate spline
3. Produce and display a plot showing the smoothed tracks.

The zeroing and smoothing stages both produce CSV files with the results. You 
can take these and open them in Excel or LibreOffice (or your preferred analysis
environment) and do whatever analysis/plotting you want. These files are
in the project directory, named `zeroed_tracks.csv` and `smoothed_tracks.csv` 
respectively

The plot is only used to see what the software has produced (i.e. to check that
the software is working as expected), you are expected to produce your own 'nice' 
plots as required.

## Miscellany
### Params file
When you run the software it should create a file called 'params.json' in the software directory. Do not modify this file by hand. It stores the path of the current project and path to the project file which are relied on internally.

This allows the software to remember what the last project was when you start the software.

### Project files
Inside your project directories, a project file will be created (*.dt2p). This will store all of the information about the project (which video files to use, calibration board configuration, etc.). Again, do not modify this by hand.
