import tkinter as tk

class ToolFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init(parent, kwargs)

        self.__labelframe = tk.LabelFrame(self, text='Tools')
        self.__btn_run = tk.Button(self.__labelframe, text='Run')
        self.__btn_quit = tk.Button(self.__labelframe,text='Quit')

        # Tool radio buttons
        self.__rbn_var = tk.IntVar(self, 0)
        self.__rbn_configure_calibration =\
              tk.Radiobutton(self.__labelframe,
                             text='1. Configure calibration board')
        self.__rbn_compute_calibration =\
              tk.Radiobutton(self.__labelframe,
                             text='2. Compute camera calibration')
        self.__rbn_autotracker =\
              tk.Radiobutton(self.__labelframe,
                             text='3. Run autotracker')
        self.__rbn_process_tracks =\
              tk.Radiobutton(self.__labelframe,
                             text='4. Process tracks (zero, calibrate, smooth)')
        
        n_rows = 5
        for i in range(n_rows):
            self.__labelframe.rowconfigure(i, weight=1)

        self.__labelframe.grid(0,0)