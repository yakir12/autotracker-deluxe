# Autotracker Deluxe

This repo is for the 'Autotracker deluxe' project wherein we are
trying to get a viable autotracking solution up and running for the
dung beetle group. The current version is untested.

## Installation
### Prerequisites

This software depends on python3 and pip. You must have these
installed to run the software. 

### Download

1. Click the 'Code' dropdown above.
2. Click 'Download ZIP'
3. Extract the Downloaded zip file and put the folder containing the code somewhere sensible.

You can also clone the repo using Git if you know how to do this.

### Opening a terminal in the right place

In both cases, you can navigate to the code directory using the
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

**Important! :** The tracker depends on the `opencv-contrib-python` package. If you already have a version of opencv installed via pip then you either need to uninstall that version of OpenCV or set up a virtual environment. The `opencv-contrib-python` package should subsume the standard `opencv-python` package so the switch shouldn't cause problems with any other software.

In the terminal, type:

`pip install -r requirements.txt`

and press enter. You should only need to do this the first time you download the software.

## Usage instructions

Open the code folder in the terminal as described above.

### Setup
1. In the terminal, type `python main.py` and press enter. A window should open requesting user information.
2. Enter a Username and Session ID. For example 'Elin' and '2023-02-09_beetle01'.
3. Click Submit, then close the window.
4. A new window should open asking you to select a calibraiton video. Click 'Browse'.
5. Find the calibration video you want to use. Click 'Open', then click 'Upload file', then close the window.
6. A new window should open asking you to select a raw video for tracking.
Click 'Browse'.
7. Find the video you want to track. Click 'Open', then click 'Upload file', then close the window.
8. A new window should open for offline tracking. Follow the on-screen instructions. If you encounter any errors, see the previous section.

**Note:** The vision component of the software can be used but the analysis portion is unavailable at present. This means you can upload video files and a calibration file, calibrate your video, and autotrack the beetle; but you cannot analyse/plot a track.
