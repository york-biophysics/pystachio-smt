# PySTACHIO: Quick-Start Guide

## Installation

The easiest way to install PySTACHIO is by using pip - but note the package name is pystachio-smt and will only work with the Python 3 version of pip. As usual, pip will automatially resolve, download, and install dependencies.

If you prefer not to use pip for some reason, you can install manually. Requirements are found in requirements.txt in the PySTACHIO directory. There is no Makefile for PySTACHIO so after unpacking the download and installing dependencies you can start straight away.

## Running PySTACHIO

PySTACHIO has two user modes: GUI via apps built on matplotlib and Tkinter and a command line interface that is suitable for use with Linux, Mac, or a Python interpreter in Windows. Probably it works OK with PowerShell too but we haven't tried it because none of us know PowerShell. In general we believe that the command line interface is the best, most flexible way to use PySTACHIO. The GUIs are useful for testing analysis parameters and visualizing PySTACHIO's performance and output, but for a large data set or convergence testing it will almost always be necessary to write a script to call PySTACHIO recursively with modified parameters or different data files, unless you like clicking the same few buttons every few minutes for an entire day.

### From the GUI

To get up and running with a GUI, run it:

   $python3 gui.py

This will start the GUI, ask you for to pick a file, and load up a matplotlib window with clicky buttons and changeable parameters. You can use the "find spots" button to see where PySTACHIO has found spots in the displayed frame of your image. When you are satisfied that the parameters are OK for your data, you can click "run full analysis" to run the full analysis on the whole image stack.

To use 'click mode', run it:

   $python3 clickmode.py

You can click a spot you like the look of, and the centre will be refined as usual for PySTACHIO, and the intensity of the spot tracked through time and displayed. You can click as many spots as you like, clear them at any time, or choose a new file to work on instead. This mode is primarily useful to see "overtracking" and stepwise photobleaching of molecular clusters or assemblies.

To use the smFRET code, run it:

   $python3 smFRET.py

This is still a work in progress.

### From the command line

PySTACHIO on the command line has a relatively straightforward syntax which should be trivially scriptable (here square brackets indicate a required argument and curly braces indicate optional arguments):

    $python3 pystachio-smt [TASK_LIST] [IMAGE_FILE] {KEYWORD_ARGUMENTS}

We will take each of these in order:

**TASK_LIST** This is a list of tasks for PySTACHIO to execute and is made up of one or more tasks taken from simulate track postprocess view app where each task is separated by a comma but no space, e.g. track,postprocess. Note here that order matters! Tasks are executed left to right so simulate,track is different to track,simulate. In the former, you simulate data and then track it. In the latter, you look for a file, track it if it exists (crash if it doesn't) and then simulate data __and overwrite whatever it was you just tracked_. Beware! If you run ``pystachio-smt app`` you launch the web app and can navigate to `localhost:8050` to use the GUI - this is in fact all the clickable shortcuts do. 

**IMAGE_FILE** This is the path (full or relative) to the image file for analysis. If you are simulating data, this specifies the location and root filename for the simulated data to be saved to. You should specify the filename without the .tif extension. PySTACHIO at this time only supports TIF files.

**KEYWORD\_ARGUMENTS** These are optional and overwrite the defaults found in parameters.py. To use them, specify the keyword and a new value separated by = e.g.: `SNR_min=0.6`. Multiple keyword arguments should be separated by spaces. For a full list of keyword arguments see the manual.

For tracking or postprocessing, analysis files will be written to the same directory as the image file is found, as will graphs of the analysis. Probably you will need to replot the data but we hope that the default generated files are helpful at least as an indication or to begin more involved analysis.