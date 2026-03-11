#git clone 

import tkinter as tk
import serial
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression

# ==========================================
# CONFIGURATION
# ==========================================
SERIAL_PORT = 'COM16'  # Set to COM16 based on your Arduino setup
BAUD_RATE = 9600
MIN_DATA_POINTS = 15   # Minimum points needed before ML training starts
WINDOW_SIZE = 20       # How many recent points the ML model uses to train
PLOT_HISTORY_LIMIT = 50 # How many points to show on the screen at once

# Global arrays to store data
history_x = []
history_y = []
predicted_x = []
predicted_y = []

current_time = 0  # Represents our x-axis time step

# ==========================================
# PHASE 1: GUI Setup & Initial Plot
# ==========================================
root = tk.Tk()
root.title("Smart System ML Integration - Real-Time Predictive GUI")
root.geometry("800x600")

# Create a Matplotlib Figure and Axis
fig = Figure(figsize=(8, 5), dpi=100)
ax = fig.add_subplot(111)

# Draw an initial empty grid so the window isn't completely blank on startup
ax.set_title("Waiting for Arduino Data on " + SERIAL_PORT + "...")
ax.set_xlabel("Time (Data Points)")
ax.set_ylabel("Sensor Value")
ax.grid(True)

# Embed the Matplotlib Figure into the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas.draw()

# Connect to Serial Port
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    print(f"Successfully connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error connecting to {SERIAL_PORT}: {e}")
    print("Make sure your Arduino is plugged in and the Arduino IDE Serial Monitor is CLOSED.")

# ==========================================
# PHASE 2 & 3: Live Reading, ML, and Plotting
# ==========================================
def update_system():
    global current_time
    
    # 1. Read live sensor data (Non-blocking check)
    if 'ser' in globals() and ser.in_waiting > 0:
        try:
            # Read exactly what the Arduino sends
            raw_line = ser.readline()
            line = raw_line.decode('utf-8').strip()
            
            if line:
                print(f"Arduino sent: '{line}'") # DEBUG: See what is coming in
                
                # Convert the text to a decimal number
                current_val = float(line)
                
                # Update memory arrays
                history_x.append(current_time)
                history_y.append(current_val)
                
                # --- Phase 2: Dynamic Model Training & Prediction ---
                # Only train if we have collected enough data
                if len(history_y) >= MIN_DATA_POINTS:
                    
                    # Implement sliding window (get the last WINDOW_SIZE points)
                    X_train = np.array(history_x[-WINDOW_SIZE:]).reshape(-1, 1)
                    y_train = np.array(history_y[-WINDOW_SIZE:])
                    
                    # Train the Linear Regression model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Predict the next upcoming sensor value
                    next_time = current_time + 1
                    pred_val = model.predict([[next_time]])[0]
                    
                    # Save predictions for plotting
                    predicted_x.append(next_time)
                    predicted_y.append(pred_val)
                
                # --- Phase 3: Real-Time Visualization ---
                ax.clear() # Clear old graph
                
                # Keep only the most recent data on the plot to prevent overcrowding
                display_x = history_x[-PLOT_HISTORY_LIMIT:]
                display_y = history_y[-PLOT_HISTORY_LIMIT:]
                
                # Plot actual collected sensor data (Solid blue line with dots)
                ax.plot(display_x, display_y, marker='o', color='blue', linestyle='-', label="Actual Sensor Data")
                
                # Plot the machine learning model's predictions
                if len(predicted_x) > 0:
                    disp_pred_x = predicted_x[-PLOT_HISTORY_LIMIT:]
                    disp_pred_y = predicted_y[-PLOT_HISTORY_LIMIT:]
                    
                    # Plot historical predictions as a dashed red line
                    ax.plot(disp_pred_x, disp_pred_y, color='red', linestyle='--', label="ML Trend Line")
                    
                    # Highlight the VERY NEXT predicted future value (distinct red star)
                    ax.plot([predicted_x[-1]], [predicted_y[-1]], marker='*', color='red', markersize=12, label="Next Prediction")
                
                # Formatting the plot
                ax.set_title(f"Live Arduino Data & ML Prediction (Port: {SERIAL_PORT})")
                ax.set_xlabel("Time (Data Points)")
                ax.set_ylabel("Sensor Value")
                ax.legend(loc="upper left")
                ax.grid(True)
                
                # Redraw the canvas
                canvas.draw()
                
                # Increment time step
                current_time += 1
                
        except ValueError:
            print(f"WARNING: Could not convert '{line}' to a number. Make sure Arduino is only sending numbers!")
        except Exception as e:
            # Ignore random garbled bytes during the connection phase
            pass

    # Phase 3: Non-Blocking Loop
    # Schedule this function to run again in 50 milliseconds
    root.after(50, update_system)

# Start the non-blocking update loop
update_system()



# Start the Tkinter main graphical loop
root.mainloop()
