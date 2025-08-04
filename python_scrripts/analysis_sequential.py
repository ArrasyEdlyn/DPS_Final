import pandas as pd
import numpy as np
import time
import os
import psutil 
import platform


def load_and_prepare_data(filepath='train.csv'):
    """
    This function loads the dataset from a CSV file, displays basic info,
    and returns the 'trip_duration' column as a list.
    """
    print(f"Attempting to load data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        print("Please make sure you have downloaded the 'train.csv' dataset from Kaggle")
        print("and placed it in the same folder as this script.")
        return None

    try:
        
        df = pd.read_csv(filepath)

        print("\n--- Initial Dataset Information ---")
       
        trip_durations = pd.to_numeric(df['trip_duration'], errors='coerce').dropna()

        
        print(f"\nData count before cleaning: {len(trip_durations)}")
        trip_durations = trip_durations[(trip_durations > 60) & (trip_durations < 21600)]
        print(f"Data count after cleaning: {len(trip_durations)}")

        data_list = trip_durations.tolist()
        print(f"\nSuccessfully loaded and cleaned {len(data_list)} trip duration records.")
        return data_list

    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None


def get_machine_specs():
    """
    Function to get and display CPU and RAM specifications.
    """
    print("\n--- Machine Specifications ---")
    system = platform.system()
    try:
        if system == "Windows":
            processor = platform.processor()
        elif system == "Darwin": # macOS
            processor = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
        elif system == "Linux":
            processor_info = os.popen('cat /proc/cpuinfo | grep "model name" | uniq').read().strip()
            processor = processor_info.split(':')[1].strip()
        else:
            processor = "N/A"
    except Exception:
        processor = platform.processor() # Fallback

    print(f"Processor: {processor}")
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"Total RAM: {ram_gb:.2f} GB")
    print("--------------------------\n")



def sequential_sort(data_slice):
    """
    Sorts the data using a single process.
    We create a copy to ensure the original data_slice is not modified.
    """
    return sorted(data_slice.copy())

def sequential_filter(data_slice, threshold):
    """
    Filters the data using a single process (list comprehension).
    """
    return [d for d in data_slice if d > threshold]



if __name__ == "__main__":
    
    trip_data = load_and_prepare_data('train.csv')

    if trip_data:
       
        data_splits = [0.25, 0.50, 0.75, 1.0]
        filter_threshold = 1000 
        results = {} 

        print("\n--- Starting Sequential Analysis ---")

        for split in data_splits:
            slice_size = int(len(trip_data) * split)
            data_slice = trip_data[:slice_size]
            split_label = f"{int(split*100)}%"
            results[split_label] = {}
            
            print(f"\n--- Processing {split_label} of data ({slice_size} records) ---")

            
            start_time = time.time()
            sorted_data = sequential_sort(data_slice)
            end_time = time.time()
            duration = end_time - start_time
            results[split_label]['Sequential Sort'] = duration
            print(f"Sequential Sort took: {duration:.4f} seconds.")
            

            start_time = time.time()
            filtered_data = sequential_filter(data_slice, filter_threshold)
            end_time = time.time()
            duration = end_time - start_time
            results[split_label]['Sequential Filter'] = duration
            print(f"Sequential Filter took: {duration:.4f} seconds. Found {len(filtered_data)} records > {filter_threshold}.")

        
        print("\n\n--- Summary of Sequential Performance ---")
        
        results_df = pd.DataFrame(results).T 
        print(results_df)
        print("\nNext step: Implement Thread and Multiprocessing analysis.")

