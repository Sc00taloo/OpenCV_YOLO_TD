import tkinter as tk
from Kanny import tracking, algo
from tkinter import messagebox

DEFAULT_VALUES = [3, 2, 50, 100]
entries = []

def run_kanny():
    inputs = []
    method = 'Yes'
    for entry, default in zip(entries, DEFAULT_VALUES):
        value = entry.get()
        if value.strip() == "":
            inputs.append(default)
        else:
            try:
                inputs.append(round(float(value)))
            except ValueError:
                messagebox.showerror("Ошибка", f"'{value}' не является числом!")
                return
    tracking(inputs, method)

def run_not_sobel():
    inputs = []
    method = 'Not'
    for entry, default in zip(entries, DEFAULT_VALUES):
        value = entry.get()
        if value.strip() == "":
            inputs.append(default)
        else:
            try:
                inputs.append(round(float(value)))
            except ValueError:
                messagebox.showerror("Ошибка", f"'{value}' не является числом!")
                return
    tracking(inputs, method)

def run_not_not_sobel():
    inputs = []
    method = 'NotNot'
    for entry, default in zip(entries, DEFAULT_VALUES):
        value = entry.get()
        if value.strip() == "":
            inputs.append(default)
        else:
            try:
                inputs.append(round(float(value)))
            except ValueError:
                messagebox.showerror("Ошибка", f"'{value}' не является числом!")
                return
    tracking(inputs, method)

def run_both():
    inputs = []
    method = 'Both'
    for entry, default in zip(entries, DEFAULT_VALUES):
        value = entry.get()
        if value.strip() == "":
            inputs.append(default)
        else:
            try:
                inputs.append(round(float(value)))
            except ValueError:
                messagebox.showerror("Ошибка", f"'{value}' не является числом!")
                return
    tracking(inputs, method)

def run_new_algo():
    inputs = []
    for entry, default in zip(entries, DEFAULT_VALUES):
        value = entry.get()
        if value.strip() == "":
            inputs.append(default)
        else:
            try:
                inputs.append(round(float(value)))
            except ValueError:
                messagebox.showerror("Ошибка", f"'{value}' не является числом!")
                return
    algo(inputs)

def run_my_algo():
    inputs = []
    for entry, default in zip(entries, DEFAULT_VALUES):
        value = entry.get()
        if value.strip() == "":
            inputs.append(default)
        else:
            try:
                inputs.append(round(float(value)))
            except ValueError:
                messagebox.showerror("Ошибка", f"'{value}' не является числом!")
                return
    algo(inputs)

def create_menu():
    root = tk.Tk()
    root.title("Выбор метода")

    frame = tk.Frame(root)
    frame.pack(pady=5)
    label = tk.Label(frame, text="Ядро:")
    label.pack(side=tk.LEFT)
    entry = tk.Entry(frame, validate="key")
    entry.pack(side=tk.LEFT)
    entries.append(entry)
    entry.config(validate="key", validatecommand=(root.register(lambda val: val.isdigit() or val == ""), "%P"))

    frame = tk.Frame(root)
    frame.pack(pady=5)
    label = tk.Label(frame, text="Сигма:")
    label.pack(side=tk.LEFT)
    entry = tk.Entry(frame, validate="key")
    entry.pack(side=tk.LEFT)
    entries.append(entry)
    entry.config(validate="key", validatecommand=(root.register(lambda val: val.isdigit() or val == ""), "%P"))

    frame = tk.Frame(root)
    frame.pack(pady=5)
    label = tk.Label(frame, text="Нижний предел:")
    label.pack(side=tk.LEFT)
    entry = tk.Entry(frame, validate="key")
    entry.pack(side=tk.LEFT)
    entries.append(entry)
    entry.config(validate="key", validatecommand=(root.register(lambda val: val.isdigit() or val == ""), "%P"))

    frame = tk.Frame(root)
    frame.pack(pady=5)
    label = tk.Label(frame, text="Верхний предел:")
    label.pack(side=tk.LEFT)
    entry = tk.Entry(frame, validate="key")
    entry.pack(side=tk.LEFT)
    entries.append(entry)
    entry.config(validate="key", validatecommand=(root.register(lambda val: val.isdigit() or val == ""), "%P"))

    # Создаем кнопку
    button = tk.Button(root, text="Kanny_Sobel", command=run_kanny)
    button.pack(pady=10)

    button1 = tk.Button(root, text="Kanny_not_Sobel", command=run_not_sobel)
    button1.pack(pady=10)

    button5 = tk.Button(root, text="Kanny_not_not_Sobel", command=run_not_not_sobel)
    button5.pack(pady=10)

    button2 = tk.Button(root, text="Both", command=run_both)
    button2.pack(pady=10)

    button3 = tk.Button(root, text="Algo", command=run_new_algo)
    button3.pack(pady=10)

    button4 = tk.Button(root, text="My algo", command=run_my_algo)
    button4.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_menu()