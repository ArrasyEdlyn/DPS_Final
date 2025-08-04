import pandas as pd
import numpy as np
import time
import os
import psutil 
import platform
import multiprocessing
from functools import partial
import heapq



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
        elif system == "Darwin": 
            processor = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
        elif system == "Linux":
            processor_info = os.popen('cat /proc/cpuinfo | grep "model name" | uniq').read().strip()
            processor = processor_info.split(':')[1].strip()
        else:
            processor = "N/A"
    except Exception:
        processor = platform.processor() 

    print(f"Processor: {processor}")
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"Total RAM: {ram_gb:.2f} GB")
    num_cores = os.cpu_count()
    print(f"CPU Cores: {num_cores}")
    print("--------------------------\n")
    return num_cores


def sequential_sort(data_slice):
    """Sorts the data using a single process."""
    return sorted(data_slice.copy())

def sequential_filter(data_slice, threshold):
    """Filters the data using a single process."""
    return [d for d in data_slice if d > threshold]


def sort_chunk(chunk):
    """Helper function to sort a single chunk of data."""
    return sorted(chunk)

def filter_chunk(chunk, threshold):
    """Helper function to filter a single chunk of data."""
    return [d for d in chunk if d > threshold]


def parallel_sort(data_slice, num_processes):
    """
    Sorts the data in parallel using a divide-and-conquer approach.
    1. Splits data into chunks.
    2. Sorts each chunk in a separate process.
    3. Merges the sorted chunks back together.
    """
    pool = multiprocessing.Pool(processes=num_processes)
    
    chunk_size = int(np.ceil(len(data_slice) / num_processes))
    chunks = [data_slice[i:i + chunk_size] for i in range(0, len(data_slice), chunk_size)]
    
    
    sorted_chunks = pool.map(sort_chunk, chunks)
    
    pool.close()
    pool.join()
    
    
    return list(heapq.merge(*sorted_chunks))

def parallel_filter(data_slice, threshold, num_processes):
    """
    Filters the data in parallel.
    1. Splits data into chunks.
    2. Filters each chunk in a separate process.
    3. Combines the filtered results.
    """
    pool = multiprocessing.Pool(processes=num_processes)
    chunk_size = int(np.ceil(len(data_slice) / num_processes))
    chunks = [data_slice[i:i + chunk_size] for i in range(0, len(data_slice), chunk_size)]
    
    
    filter_with_threshold = partial(filter_chunk, threshold=threshold)
    
    
    filtered_chunks = pool.map(filter_with_threshold, chunks)
    
    pool.close()
    pool.join()
    
    
    return [item for sublist in filtered_chunks for item in sublist]




if __name__ == "__main__":
   
    num_cores = get_machine_specs()
    trip_data = load_and_prepare_data('train.csv')

    if trip_data:
        data_splits = [0.25, 0.50, 0.75, 1.0]
        filter_threshold = 1000
        results = {}

        for split in data_splits:
            slice_size = int(len(trip_data) * split)
            data_slice = trip_data[:slice_size]
            split_label = f"{int(split*100)}%"
            results[split_label] = {}
            
            print(f"\n--- Processing {split_label} of data ({slice_size} records) ---")

            
            start_time = time.time()
            seq_sorted = sequential_sort(data_slice)
            results[split_label]['Sequential Sort'] = time.time() - start_time
            print(f"Sequential Sort took: {results[split_label]['Sequential Sort']:.4f} seconds.")

            start_time = time.time()
            seq_filtered = sequential_filter(data_slice, filter_threshold)
            results[split_label]['Sequential Filter'] = time.time() - start_time
            print(f"Sequential Filter took: {results[split_label]['Sequential Filter']:.4f} seconds.")

            
            start_time = time.time()
            par_sorted = parallel_sort(data_slice, num_cores)
            results[split_label]['Multiprocessing Sort'] = time.time() - start_time
            print(f"Multiprocessing Sort took: {results[split_label]['Multiprocessing Sort']:.4f} seconds.")
            

            start_time = time.time()
            par_filtered = parallel_filter(data_slice, filter_threshold, num_cores)
            results[split_label]['Multiprocessing Filter'] = time.time() - start_time
            print(f"Multiprocessing Filter took: {results[split_label]['Multiprocessing Filter']:.4f} seconds.")
            


        
        print("\n\n--- Summary of Performance ---")
        results_df = pd.DataFrame(results).T
        print(results_df)
        print("\nNext step: Implement Threading analysis and finalize the report.")
