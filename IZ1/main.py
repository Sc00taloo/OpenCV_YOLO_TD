import tkinter as tk
from CSRT import csrt_tracking
from MedianFlow import medianFlow_tracking
from MOSSE import mosse_tracking
from Summary_table import sumTable

def run_csrt():
    csrt_tracking()
def run_medianflow():
    medianFlow_tracking()
def run_mosse():
    mosse_tracking()
def sum_table():
    sumTable()

def create_menu():
    root = tk.Tk()
    root.title("Выбор метода")

    button_mosse = tk.Button(root, text="MOSSE", command=run_mosse)
    button_mosse.pack(pady=10)

    button_medianflow = tk.Button(root, text="MedianFlow", command=run_medianflow)
    button_medianflow.pack(pady=10)

    button_csrt = tk.Button(root, text="CSRT", command=run_csrt)
    button_csrt.pack(pady=10)

    button_csrt = tk.Button(root, text="Summary Table", command=sum_table)
    button_csrt.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_menu()