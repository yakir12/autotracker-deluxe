import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np

from old_calibration import make_checkerboard
from project import project_file

class ChessboardSelector(tk.Toplevel):
    def __init__(self, parent, title='', **kwargs):
        super().__init__(parent, **kwargs)

        self.title("Confgure chessboard pattern")
        self.minsize(500, 500)

        # Window grid configuration
        window_n_cols = 1
        window_n_rows = 6
        for i in range(window_n_cols):
            for j in range(window_n_rows):
                row_weight = 1 if j < 2 or j > 4 else 2
                self.columnconfigure(i, weight=1)
                self.rowconfigure(j, weight=row_weight)

        # Window children
        self.__lbf_dims_selector = tk.LabelFrame(self, 
                                                 text='Select checkerboard dimensions')
        self.__lbf_chessboard_example = tk.LabelFrame(self,
                                                      text='Target pattern')
        self.__frm_controls = tk.Frame(self)

        # Window population
        self.__lbf_dims_selector.grid(row=0, column=0, sticky='new')
        self.__lbf_chessboard_example.grid(row=1, column=0, rowspan=4, sticky='nesw')
        self.__frm_controls.grid(row=5, column=0, sticky='esw')


        # Dimension selector frame geometry configuration    
        self.__lbf_dims_selector.columnconfigure(0, weight=1)
        self.__lbf_dims_selector.columnconfigure(1, weight=1)
        
        # Dimension selector children
        self.__chessboard_min = 3 # Minimum for inner corner detection to make sense.
        self.__chessboard_max = 30 # Arbitrarily chosen

        self.__stv_row = tk.StringVar()
        self.__stv_col = tk.StringVar()        
        self.__stv_square_size = tk.StringVar()
        self.__spb_row_selector = tk.Spinbox(self.__lbf_dims_selector,                                             
                                             from_=self.__chessboard_min, 
                                             to=self.__chessboard_max,
                                             state="readonly",
                                             command=self.__update_checkerboard, 
                                             background='white', 
                                             textvariable=self.__stv_row)
        self.__spb_col_selector = tk.Spinbox(self.__lbf_dims_selector,
                                             from_=self.__chessboard_min, 
                                             to=self.__chessboard_max,
                                             state="readonly",
                                             command=self.__update_checkerboard, 
                                             background='white', 
                                             textvariable=self.__stv_col)
        
        self.__spb_sq_size_selector = tk.Spinbox(self.__lbf_dims_selector,
                                                 from_=20, 
                                                 to=60,
                                                 state="readonly",
                                                 background='white', 
                                                 textvariable=self.__stv_square_size,
                                                 command=self.__update_checkerboard)

        self.__lbl_rows = tk.Label(self.__lbf_dims_selector,
                                   text="Rows")
        self.__lbl_columns = tk.Label(self.__lbf_dims_selector, 
                                      text="Columns")    
        self.__lbl_square_size = tk.Label(self.__lbf_dims_selector,
                                          text="Square size (mm)")    
                
        
        # Dimension selector frame population
        self.__lbl_rows.grid(row=0, column=0, sticky='ew')
        self.__lbl_columns.grid(row=0, column=1, sticky='ew')
        self.__lbl_square_size.grid(row=0, column=2, sticky='ew')
        
        
        self.__spb_row_selector.grid(row=1, column=0, sticky='nesw')
        self.__spb_col_selector.grid(row=1, column=1, sticky='nesw')
        self.__spb_sq_size_selector.grid(row=1, column=2, sticky='nesw')

        # Try to populate spinboxes from project file
        try:
            n_rows = project_file["chessboard_rows"]
            n_cols = project_file["chessboard_columns"]
            square_size = project_file["chessboard_square_size"]
        except KeyError:
            print("Chessboard dimensions not found in project file, adding...")
            n_rows = 6
            n_cols = 9
            square_size = 39
            project_file["chessboard_rows"] = n_rows
            project_file["chessboard_columns"] = n_cols
            project_file["chessboard_square_size"] = square_size

        self.__stv_square_size.set(square_size)
        self.__stv_col.set(n_cols)
        self.__stv_row.set(n_rows)

        # Chessboard example frame geometry configuration
        self.__lbf_chessboard_example.columnconfigure(0, weight=1)
        self.__lbf_chessboard_example.rowconfigure(0, weight=1)
        self.__lbf_chessboard_example.rowconfigure(1, weight=0)


        # Checkerboard example children
        info_string = "This is the pattern the calibration algorithm will" +\
                      " look for! The algorithm looks for 'inner corners'" +\
                      " which are displayed over the board. The pattern should be " +\
                      " visible in full in your video (i.e. no tape" +\
                      " covering part of the square)."
        self.__cnv_chessboard = tk.Canvas(self.__lbf_chessboard_example)
        self.__lbl_info = tk.Label(self.__lbf_chessboard_example,
                                   text=info_string,
                                   wraplength=500,
                                   relief='sunken', 
                                   justify='left',
                                   anchor='w')
        
        self.__update_checkerboard()
        self.__cnv_chessboard.grid(row=0, column=0, sticky='nw', padx=(10,10), pady=(10,10))
        self.__lbl_info.grid(row=1, column=0, sticky='nesw')
        
        # Control frame children
        self.__btn_confirm = tk.Button(self.__frm_controls, 
                                       text='Confirm', 
                                       command=self.__confirm_callback)
        self.__btn_cancel = tk.Button(self.__frm_controls, 
                                      text='Cancel', 
                                      command=self.__cancel_callback)

        # Control frame population
        self.__btn_confirm.grid(row=0, column=0, sticky='nw')
        self.__btn_cancel.grid(row=0, column=1, sticky='nw')

    def __update_checkerboard(self):
        # Make the chessboard, note that the 'square_size' parameter chosen 
        # is arbitrary. This is just for display so it doesn't matter.
        sq_size = int(self.__spb_sq_size_selector.get())# 30
        n_rows = int(self.__spb_row_selector.get())
        n_cols = int(self.__spb_col_selector.get())
        
        chessboard, chessboard_size = make_checkerboard(n_rows,
                                          n_cols,
                                          sq_size)
        chessboard *= 255 # Make '1' entries white.


        success, corners = cv2.findChessboardCorners(chessboard.astype(np.uint8), 
                                                     chessboard_size)
        chessboard = cv2.cvtColor(chessboard.astype(np.uint8), cv2.COLOR_GRAY2BGR)        
        chessboard = cv2.drawChessboardCorners(chessboard.astype(np.uint8),
                                               patternSize=chessboard_size,
                                               corners=corners, 
                                               patternWasFound=success)
        
        
        
        self.__chessboard_image = ImageTk.PhotoImage(Image.fromarray(chessboard))
        self.__cnv_chessboard.create_image(0, 0, anchor='nw', image=self.__chessboard_image)
        self.__cnv_chessboard.configure(width=n_cols*sq_size,
                                        height=n_rows*sq_size)
        
                        
    def __confirm_callback(self):
        project_file["chessboard_rows"] = int(self.__spb_row_selector.get())
        project_file["chessboard_columns"] = int(self.__spb_col_selector.get())
        project_file["chessboard_square_size"] = self.__spb_sq_size_selector.get()
        self.destroy()

    def __cancel_callback(self):
        self.destroy()
        

    
