import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import joblib
import numpy as np
import pandas as pd
import os

# Load the trained model
model = joblib.load("../src/model/wine_quality_model.pkl")

def predict_quality():
    """
    Gather input features from user entries, predict the wine quality using the model,
    and display the result in a popup window. Optionally, save the user-entered data.
    """
    try:
        # Gather input features from user entries
        features = [
            float(fixed_acidity_entry.get()),
            float(volatile_acidity_entry.get()),
            float(citric_acid_entry.get()),
            float(residual_sugar_entry.get()),
            float(chlorides_entry.get()),
            float(free_sulfur_dioxide_entry.get()),
            float(total_sulfur_dioxide_entry.get()),
            float(density_entry.get()),
            float(pH_entry.get()),
            float(sulphates_entry.get()),
            float(alcohol_entry.get())
        ]
        
        # Convert the features to a numpy array
        features = np.array([features])
        
        # Predict the wine quality
        predicted_quality = model.predict(features)[0]
        
        # Show the result in a popup window
        messagebox.showinfo("Predicted Quality", f"The predicted quality of the wine is: {predicted_quality}")
        
        # Ask the user if they want to save the entered data
        save_data = messagebox.askyesno("Save Data", "Do you want to save the entered data?")
        if save_data:
            save_user_data(features, predicted_quality)
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields")

def save_user_data(features, quality):
    """
    Save user-entered data and predicted quality to a CSV file.
    
    Args:
        features (numpy.ndarray): The input features for the prediction.
        quality (int): The predicted quality of the wine.
    """
    # Get additional details from the user
    name = simpledialog.askstring("Input", "Enter the name of the wine:")
    description = simpledialog.askstring("Input", "Enter a description for the wine:")
    
    # Create a dictionary of the data
    data = {
        "Name": name,
        "Description": description,
        "Fixed Acidity": features[0][0],
        "Volatile Acidity": features[0][1],
        "Citric Acid": features[0][2],
        "Residual Sugar": features[0][3],
        "Chlorides": features[0][4],
        "Free Sulfur Dioxide": features[0][5],
        "Total Sulfur Dioxide": features[0][6],
        "Density": features[0][7],
        "pH": features[0][8],
        "Sulphates": features[0][9],
        "Alcohol": features[0][10],
        "Quality": quality
    }
    
    # Convert the data to a DataFrame and save it to a CSV file
    df = pd.DataFrame([data])
    file_path = "../data/user_data.csv"
    
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)

def open_prediction_window():
    """
    Open a new window with entry fields for the user to input wine features
    and a button to predict wine quality.
    """
    prediction_window = tk.Toplevel(root)
    prediction_window.title("Wine Quality Prediction")
    prediction_window.geometry("400x600")
    prediction_window.configure(bg='#f0f0f0')

    # List of feature labels
    labels = [
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
        "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density",
        "pH", "Sulphates", "Alcohol"
    ]
    
    global entries
    entries = []

    # Create entry fields for each feature
    for i, label_text in enumerate(labels):
        label = ttk.Label(prediction_window, text=label_text, background='#f0f0f0')
        label.grid(row=i, column=0, padx=10, pady=5, sticky='E')

        entry = ttk.Entry(prediction_window)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky='W')
        entries.append(entry)

    # Assign entries to global variables
    global fixed_acidity_entry, volatile_acidity_entry, citric_acid_entry
    global residual_sugar_entry, chlorides_entry, free_sulfur_dioxide_entry
    global total_sulfur_dioxide_entry, density_entry, pH_entry
    global sulphates_entry, alcohol_entry

    (fixed_acidity_entry, volatile_acidity_entry, citric_acid_entry,
     residual_sugar_entry, chlorides_entry, free_sulfur_dioxide_entry,
     total_sulfur_dioxide_entry, density_entry, pH_entry,
     sulphates_entry, alcohol_entry) = entries

    # Button to predict wine quality
    predict_button = ttk.Button(prediction_window, text="Predict Quality", command=predict_quality)
    predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

def show_inventory():
    """
    Open a new window to display the wine inventory dataset in a Treeview.
    """
    inventory_window = tk.Toplevel(root)
    inventory_window.title("Wine Inventory")
    inventory_window.geometry("800x600")
    inventory_window.configure(bg='#f0f0f0')

    # Load the inventory data
    df = pd.read_csv("../data/inventory.csv")

    # Create a Treeview to display the data
    tree = ttk.Treeview(inventory_window)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    # Add headings to the Treeview
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Add data to the Treeview
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill='both')

def show_user_wines():
    """
    Open a new window to display user-created wines in a Treeview.
    """
    user_wines_window = tk.Toplevel(root)
    user_wines_window.title("User Created Wines")
    user_wines_window.geometry("800x600")
    user_wines_window.configure(bg='#f0f0f0')

    # Load user data if it exists, otherwise create an empty DataFrame
    file_path = "../data/user_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Name", "Description", "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density", "pH", "Sulphates", "Alcohol", "Quality"])
    
    # Create a Treeview to display the data
    tree = ttk.Treeview(user_wines_window)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    # Add headings to the Treeview
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Add data to the Treeview
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill='both')

def show_main_menu():
    """
    Display the main menu with options to predict wine quality, show wine inventory,
    show user-created wines, and exit the application.
    """
    root.title("Wine Quality Project")
    root.geometry("400x400")
    root.configure(bg='#f0f0f0')
    
    # Main menu label
    label = tk.Label(root, text="Wine Quality Project", font=("Helvetica", 20, "bold"), bg='#f0f0f0')
    label.pack(pady=20)
    
    # Button to open the prediction window
    predict_button = ttk.Button(root, text="Predict Wine Quality", command=open_prediction_window)
    predict_button.pack(pady=10)
    
    # Button to show the wine inventory
    inventory_button = ttk.Button(root, text="Show Wine Inventory", command=show_inventory)
    inventory_button.pack(pady=10)

    # Button to show user-created wines
    user_wines_button = ttk.Button(root, text="Show User Created Wines", command=show_user_wines)
    user_wines_button.pack(pady=10)

    # Button to exit the application
    exit_button = ttk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)

# Create the main window and show the main menu
root = tk.Tk()
show_main_menu()

# Run the application
root.mainloop()
