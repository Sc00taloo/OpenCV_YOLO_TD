import tkinter as tk
from Tracking import tracking
from Summary_table import sumTable

def run_csrt():
    method = "CSRT"
    tracking(method)
def run_medianflow():
    method = "MedianFlow"
    tracking(method)
def run_mosse():
    method = "MOSSE"
    tracking(method)
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