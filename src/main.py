import tkinter as tk
from view.main_window import MainWindow
from controller.app_controller import AppController

def main():
    root = tk.Tk()
    controller = AppController(None)
    view = MainWindow(root, controller)
    controller.view = view
    root.mainloop()

if __name__ == "__main__":
    main()
