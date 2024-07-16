"""
wrapped_label.py

Provides a (limited) label implementation which allows text re-wrap as the 
parent widget is resized.

Tkinter does not provide a way to have multi-line labels which wrap dynamically
as the window resizes. This could perhaps be improved by using a readonly 
Entry widget.
"""

import tkinter as tk

class WrappedLabelFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        #self.config(background='red')

        # Inner label is exposed for ease
        self.__text = tk.StringVar()
        self.__label = tk.Label(self, 
                                textvariable=self.__text,
                                wraplength=500,
                                justify=tk.LEFT,
                                anchor='nw')
        
        self.columnconfigure(0, weight=1, minsize=400)
        self.rowconfigure(0, weight=1, minsize=150)

        self.__label.columnconfigure(0, weight=1)
        self.__label.rowconfigure(0,weight=1)

        self.__label.grid(column=0, row=0, sticky='nesw')

        # Initiate wrap length based on widget size
        self.__configure_wrap_length(None)

        # Bind configuration of this frame to update wrap length on the label.
        self.bind('<Configure>', self.__configure_wrap_length)

    def __configure_wrap_length(self, event):
        self.__label.configure(wraplength=self.winfo_width() - 50)
    def set_text(self, text):
        self.__text.set(text)