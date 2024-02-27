from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import timedelta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def initialize_model(model_path):
    return YOLO(model_path)

def get_frame(sec, video):
    video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    has_frames, image = video.read()
    return has_frames, image

def draw(frame, bbox):
    x, y, w, h = map(int, bbox)
    nest_box = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2, 1)
    return nest_box

def logCrossing(tracking_id, crossingType, crossingTimestamp, all_crossings_df):
    all_crossings_df.append({'Hornet_ID': tracking_id, 'Crossing': crossingType, 'Timestamp': crossingTimestamp})
    return all_crossings_df

def countCrossing(all_crossings_df):
    all_entries = all_crossings_df[all_crossings_df['Crossing'] == 'ENTRY']
    all_exits = all_crossings_df[all_crossings_df['Crossing'] == 'EXIT']
    return len(all_entries) - len(all_exits)

def addLog(df, crossing_type, no_of_crossings, crossing_timestamp, tracker_id):
    row = {'CROSSING_TYPE': crossing_type, 
           'NO_OF_CROSSINGS': no_of_crossings, 
           'CROSSING_TIMESTAMP': crossing_timestamp,
           'TOTAL_TRACKED': tracker_id}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

def track_hornets(bbox, model, vidcap, video_source):
    x, y, w, h = bbox

    SQUARE_TOP_LEFT = sv.Point(x, y)
    SQUARE_TOP_RIGHT = sv.Point(x + w, y)
    SQUARE_BOTTOM_LEFT = sv.Point(x, y + h)
    SQUARE_BOTTOM_RIGHT = sv.Point(x + w, y + h)

    square_top = sv.LineZone(start=SQUARE_TOP_RIGHT, end=SQUARE_TOP_LEFT)
    square_bottom = sv.LineZone(start=SQUARE_BOTTOM_LEFT, end=SQUARE_BOTTOM_RIGHT)
    square_left = sv.LineZone(start=SQUARE_TOP_LEFT, end=SQUARE_BOTTOM_LEFT)
    square_right = sv.LineZone(start=SQUARE_BOTTOM_RIGHT, end=SQUARE_TOP_RIGHT)

    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # Initialise counts and timestamps (HWN 18/02/2024)
    total_in_count = 0
    total_out_count = 0
    df = pd.DataFrame()
    # startTime = time.time()
    frame_no = 0

    # Get frame rate (HWN 22/02/2024)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    for result in model.track(source=video_source, show=True, stream=True, agnostic_nms=True, persist=True):

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = detections[detections.confidence > 0.1]

        # Define timestamp in seconds (HWN 18/02/2024)
        frame_no += 1
        time_sec = frame_no / fps

        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        square_top.trigger(detections=detections)
        square_bottom.trigger(detections=detections)
        square_left.trigger(detections=detections)
        square_right.trigger(detections=detections)

        # Annotate the lines on the frame
        line_annotator.annotate(frame=frame, line_counter=square_top)
        line_annotator.annotate(frame=frame, line_counter=square_bottom)
        line_annotator.annotate(frame=frame, line_counter=square_left)
        line_annotator.annotate(frame=frame, line_counter=square_right)

        curr_total_in_count = square_top.in_count + square_bottom.in_count + square_left.in_count + square_right.in_count
        curr_total_out_count = square_top.out_count + square_bottom.out_count + square_left.out_count + square_right.out_count

        crossing_time = time_sec

        if curr_total_in_count > total_in_count:
            no_entries = curr_total_in_count - total_in_count
            total_in_count = curr_total_in_count
            df = addLog(df, 'ENTRY', no_entries, crossing_time, detections.tracker_id[0])

        if curr_total_out_count > total_out_count:
            no_exits = curr_total_out_count - total_out_count
            total_out_count = curr_total_out_count
            df = addLog(df, 'EXIT', no_exits, crossing_time, detections.tracker_id[0])
        
        

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

    return df, bbox  

def select_model_path():
    script_directory = os.path.dirname(os.path.realpath(__file__))
    model_path = filedialog.askopenfilename(initialdir=script_directory, title="Select Model", filetypes=[("Model files", "*.pt")])
    return model_path

def select_video_source():
    script_directory = os.path.dirname(os.path.realpath(__file__))
    root = tk.Tk()
    root.withdraw()

    source_choice = tk.simpledialog.askstring("Input", "Choose source (live, webcam, video):")

    if source_choice.lower() == "live":
        video_source = tk.simpledialog.askstring("Input", "Enter live stream URL:")
    elif source_choice.lower() == "webcam":
        video_source = 0 
    elif source_choice.lower() == "video":
        video_source = filedialog.askopenfilename(initialdir=script_directory, title="Select Video", filetypes=[("Video files", "*.mov;*.mp4")])
    else:
        tk.messagebox.showerror("Error", "Invalid source choice")
        return None

    return video_source

    return video_source

def plot_combined(df, video_name, bbox, output_folder):
    # Convert 'CROSSING_TIMESTAMP' to datetime
    entry_data = df[df['CROSSING_TYPE'] == 'ENTRY']
    exit_data = df[df['CROSSING_TYPE'] == 'EXIT']
    
    # Plot cumulative time series
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)

    plt.plot(entry_data['CROSSING_TIMESTAMP'], entry_data['NO_OF_CROSSINGS'].cumsum(), label='Entry', marker='o', color='blue')
    plt.plot(exit_data['CROSSING_TIMESTAMP'], exit_data['NO_OF_CROSSINGS'].cumsum(), label='Exit', marker='o', color='green')

    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative No. of Crossings')
    plt.legend()
    plt.title('Cumulative Time Series for ENTRY and EXIT')

    # Plot crossing frequency and rolling average
    plt.subplot(2, 2, 2)
    df_copy = df.copy()  
    df_copy['CROSSING_TIMESTAMP'] = pd.to_datetime(df_copy['CROSSING_TIMESTAMP'], unit='s')
    df_copy.set_index('CROSSING_TIMESTAMP', inplace=True)

    entry_data_copy = entry_data.copy()  
    entry_data_copy['CROSSING_TIMESTAMP'] = pd.to_datetime(entry_data_copy['CROSSING_TIMESTAMP'], unit='s')
    entry_data_copy.set_index('CROSSING_TIMESTAMP', inplace=True)
        
    crossing_frequency_entry = entry_data_copy['NO_OF_CROSSINGS'].resample('10S').sum()
    rolling_avg_entry = crossing_frequency_entry.rolling(window=1).mean()
    plt.plot(rolling_avg_entry.index, rolling_avg_entry, linestyle='--', color='red', label='Entry')
    
    exit_data_copy = exit_data.copy()  
    exit_data_copy['CROSSING_TIMESTAMP'] = pd.to_datetime(exit_data_copy['CROSSING_TIMESTAMP'], unit='s')
    exit_data_copy.set_index('CROSSING_TIMESTAMP', inplace=True)

    crossing_frequency_exit = exit_data_copy['NO_OF_CROSSINGS'].resample('10S').sum()
    rolling_avg_exit = crossing_frequency_exit.rolling(window=1).mean()
    plt.plot(rolling_avg_exit.index, rolling_avg_exit, linestyle='--', color='gold', label='Exit')

    crossing_frequency_total = df_copy['NO_OF_CROSSINGS'].resample('10S').sum()
    rolling_avg_total = crossing_frequency_total.rolling(window=1).mean()
    plt.plot(rolling_avg_total.index, rolling_avg_total, linestyle='--', color='darkorange', label='Total')

    plt.xlabel('Time (s)')
    plt.ylabel('Crossings per 10 seconds')
    plt.legend()
    plt.title('Crossings per 10 seconds [Rolling Average]')
        
    # Set custom tick locations
    tick_interval = pd.date_range(start=0, end=df_copy.index.max(), freq='10S')
    plt.xticks(tick_interval, (tick_interval - tick_interval.min()).total_seconds().astype(int))

    
    # Bar graph
    
    plt.subplot(2, 2, 3)

    # Metadata summary data
    total_crossings = df_copy['NO_OF_CROSSINGS'].sum()
    entry_crossings = entry_data['NO_OF_CROSSINGS'].sum()
    exit_crossings = exit_data['NO_OF_CROSSINGS'].sum()
    total_tracked_hornets = df_copy['TOTAL_TRACKED'].max()
    average_frequency_total = crossing_frequency_total.mean().round(1)
    average_frequency_entry = crossing_frequency_entry.mean().round(1)
    average_frequency_exit = crossing_frequency_exit.mean().round(1)
    video_length_timedelta = df_copy.index[-1] - df_copy.index[0]
    video_length_formatted = str(video_length_timedelta).split(".")[0]
    
    # Create a table with metadata summary
    table_data = [
        ["Number of Crossings", f"Total: {total_crossings} (Entry: {entry_crossings}, Exit: {exit_crossings})"],
        ["Number of Tracked Hornets", total_tracked_hornets],
        ["Average Crossing Frequency (/s)", f"Total: {(average_frequency_total/10).round(1)}, (Entry: {(average_frequency_entry/10).round(1)}, Exit: {(average_frequency_exit/10).round(1)})"],
        ["Tracking Time", video_length_formatted],
        ["Detection box coordinates (pixels)", f"Centre: {bbox[0]+(bbox[2]/2)}x{bbox[1]+(bbox[3]/2)}"],
        ["Detection box size (pixels)", f"Width: {bbox[2]}, Height: {bbox[3]}"]
    ]

    table = plt.table(cellText=table_data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Tracker Metadata Summary')
    plt.axis('off')
    
    # Tracked vs TOTAL Bar graph
    plt.subplot(2, 2, 4)
    plt.plot(df['CROSSING_TIMESTAMP'], range(1, len(df) + 1), label='Registered Crossings', marker='o', color='purple')
    df['TOTAL_TRACKED_STABLE'] = df['TOTAL_TRACKED'].where(df['TOTAL_TRACKED'] >= df['TOTAL_TRACKED'].shift(), df['TOTAL_TRACKED'].shift())
    plt.plot(df['CROSSING_TIMESTAMP'], df['TOTAL_TRACKED_STABLE'], label='Tracked Hornets', marker='o', color='deeppink')
    plt.xlabel('Time (s)')
    plt.ylabel('Total')
    plt.legend()
    plt.title('Cumulative Tracked Hornets vs Registered Crossings')
    
    # Save plots separately inside 'Outputs' folder
    plt.savefig(os.path.join(output_folder, f"{video_name}_combined_plots.png"))
    plt.show()
        
def main():
    # Select model and video source
    root = tk.Tk()
    root.withdraw()
    model_path = select_model_path()
    video_source = select_video_source()
    if video_source:
        video_name = os.path.basename(video_source)
    else:
        print("Video source selection canceled or invalid.")

    # Initialize model
    model = initialize_model(model_path)

    # Select bounding box
    sec = 0
    vidcap = cv2.VideoCapture(video_source)
    success, frame = get_frame(sec, vidcap)

    bbox = cv2.selectROI("Select bounding box and press SPACE", frame, showCrosshair=False)
    
    # Create 'Outputs' folder if it doesn't exist
    output_folder = os.path.join(os.path.dirname(__file__), 'Outputs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Call track_hornets within the selected bbox
    df, bbox = track_hornets(bbox, model, vidcap, video_source)

    # Release the video capture object
    vidcap.release()
    cv2.destroyAllWindows()

    # Save DataFrame and bbox CSV files inside 'Outputs' folder
    df_filename = os.path.join(output_folder, f"{video_name}.csv")
    bbox_filename = os.path.join(output_folder, f"{video_name}_bbox.csv")
    df.to_csv(df_filename, index=False)
    pd.DataFrame(bbox).to_csv(bbox_filename, index=False)

    # Plot, display, and save graphs to 'Outputs' folder
    plot_combined(df, video_name, bbox, output_folder)
    
    print("CSVs and plots saved")

if __name__ == "__main__":
    main()
    
