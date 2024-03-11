import tkinter as tk
class ProjectFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.__labelframe = tk.LabelFrame(self, text='Session info')
        self.__label = tk.Label(self.__labelframe, 
                                text='Current session')
        self.__btn_select =  tk.Button(self.__labelframe,
                                       text='Select session directory')
        self.__ent_project = tk.Entry(self.__labelframe,
                                      text='Not yet selected')

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)


        n_cols = 4
        for i in range(n_cols):
            if (i == 1) or (i==2):
                self.__labelframe.columnconfigure(i, weight=1)
                continue
            self.__labelframe.columnconfigure(i, weight=0)

        self.__labelframe.grid(row=0, column=0,sticky='ew')
        self.__label.grid(row=0, column=0, sticky='w')
        self.__ent_project.grid(row=0, column=1, columnspan=2, sticky='ew')               
        self.__btn_select.grid(row=0, column=3, sticky='e')
        