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

### Setup
1. In the terminal, run: 
   
   `$: python main.py` 
   
   A window should open requesting user information. (This may take a moment. Look for a small window.)
2. Enter a Username and Session ID. For example 'Elin' and '2023-02-09_beetle01'.
3. Click Submit, then close the window.
4. A new window should open asking you to select a calibraiton video. Click 'Browse'.
5. Find the calibration video you want to use. Click 'Open', then click 'Upload file', then close the window.
6. A new window should open asking you to select a raw video for tracking.
Click 'Browse'.
7. Find the video you want to track. Click 'Open', then click 'Upload file', then close the window.
8. A new window should open for offline tracking.

**Note:** The software currently provides beetle tracking, and track smoothing.
The software does not provide calibration for these tracks (it's implemented
but doesn't work properly at the moment). All distances are in pixels and 
if your video is significantly distorted then your tracks will be too.
Tracks are output to structured CSVs to allow you to perform your own analysis.

### Offline tracking
Each stage can be engaged independently by selecting the appropriate radio button and clicking 'Run'. The instructions below assume you have done this.

#### 0. Checkerboard type
1. Select the checkerboard option
2. Enter the checkerboard square size in mm
3. Click 'Make selection'

**Note:** For the example video provided by Elin, the checkerboard is 6 by 9 and the square size is 39mm.

#### 1. Extract calibration parameters
1. Follow the on-screen instructions.
2. When the process is finished, a calibrated frame will be shown. Press q to close the window.

**Note:** You should select only a single 'ground' frame. 

#### 2. Autotracker
1. A new window will open playing the video to be tracked (at high speed)
2. Let the video play or skip to a point you wish to start tracking
3. Press P to pause the video, then T to start a track.
4. Select a rectangle (region of interest, ROI) which contains the beetle (be generous). Press Enter and close the window.
5. Press P to play and the tracked point will be indicated by a red dot.
6. Press T to finish the current track.
7. Goto (2)

You can produce any number of tracks from the video. The tracks will be appended
to the track file (`data/<user>/<session>/raw_tracks.csv`). 

**Notes**
1. If you want to cancel your ROI selection, press C then close the window.
2. If you press T by mistake and don't want to track, press T again to cancel. If you didn't play the video while tracking was enabled, the track file won't be modified.
3. You cannot use Q to quit while the video is paused. You can just close the window.
4. Do not use the trackbar while tracking. This will probably break things.
5. The coordinates produced are the centre of mass of the 'blob' which contains
   the beetle and its ball on a given frame. 

#### 3. Calibrate and smooth tracks
This option runs in full when you click 'Run'. At present this will:
1. Zero the tracks (translate them so they all start at the same origin, (0,0)).
2. Smooth the tracks using a basic univariate spline
3. Produce and display a plot showing the smoothed tracks.

The zeroing and smoothing stages both produce CSV files with the results. You 
can take these and open them in Excel or LibreOffice (or your preferred analysis
environment) and do whatever analysis/plotting you want. These files are
in `data/<user>/<session>`, `zeroed_tracks.csv`, and `smoothed_tracks.csv` 
respectively

The plot is only used to see what the software has produced (i.e. to check that
the software is working as expected), you are expected to produce your own 'nice' 
plots as required.

#### 4. Analysis
This option ties into the older analysis code. Given that the underlying 
track storage has changed, this option is no longer compatible with tracks
produced by the new tracking stage. It is currently included for legacy
reasons but will be removed soon.

