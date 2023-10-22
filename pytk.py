import tkinter as tk
from PIL import Image, ImageTk
#import lead  # Import the lead module

def get_input_values():
    input_values = [entry_var.get() for entry_var in entry_vars]
    print("Input values:", input_values)
    return input_values

root = tk.Tk()
canvas = tk.Canvas(root, width=900, height=450)
canvas.pack()  # Use pack() instead of grid() for the canvas

# Load the original logo image
original_logo = Image.open("C:/Users/91735/Desktop/Jam fist logo.png")

# Resize the logo to your desired size (e.g., 100x50)
resized_logo = original_logo.resize((100, 50), Image.ANTIALIAS)

# Convert the resized image to a PhotoImage
logo = ImageTk.PhotoImage(resized_logo)

# Create a label and display the resized logo on the canvas
logo_label = tk.Label(canvas, image=logo)
logo_label.image = logo
logo_label.grid(row=0, column=0, padx=10, pady=10)

# Instructions
instructions = tk.Label(canvas, text="Hello, I'm here to Predict your Hot Lead", font="Raleway")
instructions.grid(column=1, row=0)

# Input Boxes and Labels
input_labels = ["Lead Origin (API/Website)",
    "Lead Source (Reference/Direct traffic)",
    "Can be reached via mail (yes/no)",
    "Can be reached via call (yes/no)",
    "Total visits (in numbers)",
    "Website approach (in numbers)",
    "Last Activity (page visited/email opened)",
    "Country",
    "Employment (unemployed/employed)",
    "City"]
entry_vars = []

# Define a custom font with bold and italic attributes
custom_font = ("Raleway", 12, "bold italic")

for i, label_text in enumerate(input_labels):
    label = tk.Label(canvas, text=label_text, font=custom_font)  # Apply custom font
    label.grid(row=i + 1, column=1, sticky='w', padx=10, pady=5)
    
    entry_var = tk.StringVar()  # Create a StringVar to store the input
    entry_vars.append(entry_var)
    
    entry = tk.Entry(canvas, textvariable=entry_var)
    entry.grid(row=i + 1, column=2, padx=10, pady=5)

# Button to get input values and make predictions
def predict_lead():
    input_values = get_input_values()
    lead.make_lead_prediction(input_values)

get_values_button = tk.Button(canvas, text="Predict", command=predict_lead)
get_values_button.grid(row=len(input_labels) + 1, column=1, columnspan=2, padx=10, pady=10)

root.mainloop()
