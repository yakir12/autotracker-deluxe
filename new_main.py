import tkinter as tk
from project_frame import ProjectFrame

window_width = 800
window_height = 400
x_padding = 10
y_padding = 0

root = tk.Tk()

# Lock frame size for ease at the moment
root.minsize(window_width,window_height)
#root.maxsize(window_width, window_height)

# Support resizability for future development
n_columns = 1
n_rows = 2
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1) 
content = tk.Frame(root)
for i in range(n_rows):
    for j in range(n_columns):
        content.columnconfigure(j, weight=1)
        content.rowconfigure(i, weight=1)

p_frame = ProjectFrame(content)

tool_frame = ProjectFrame(content)


content.grid(column=0, row=0, sticky="nesw")
p_frame.grid(row=0, column=0, sticky='new')
#tool_frame.grid(row=1, column=0, rowspan=3, sticky='ew')


root.mainloop()

