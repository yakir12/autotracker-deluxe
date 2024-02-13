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
Hold control (command) and click on the folder, then select
'Open in Terminal'.

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


## Usage instructions

Open the code folder in the terminal as you did during the installation process (either using the command line or the graphical file explorer).

### Setup
1. In the terminal, run: 
   
   `$: python main.py` 
   
   A window should open requesting user information.
2. Enter a Username and Session ID. For example 'Elin' and '2023-02-09_beetle01'.
3. Click Submit, then close the window.
4. A new window should open asking you to select a calibraiton video. Click 'Browse'.
5. Find the calibration video you want to use. Click 'Open', then click 'Upload file', then close the window.
6. A new window should open asking you to select a raw video for tracking.
Click 'Browse'.
7. Find the video you want to track. Click 'Open', then click 'Upload file', then close the window.
8. A new window should open for offline tracking.

**Note:** The vision component of the software can be used but the analysis portion is unavailable at present. This means you can upload video files and a calibration file, calibrate your video, and autotrack the beetle; but you cannot analyse/plot a track.

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

#### 2. Calibrate video
1. A new window will open playing the video to be tracked
2. Let this video play up to the end of the desired track
3. Press q to close

#### 3. Tracking
1. Use the sliders to select a start and end frame for tracking
2. Press the spacebar
3. In the new window, click and drag to form a rectangle around the beetle.
4. Press spacebar
5. Watch the tracking happen, as if by magic!

#### 4. Analysis
Not yet available

