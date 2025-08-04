# DPS_Final
This project compares filtering and sorting using sequential processing, threading, and multiprocessing on 25%, 50%, 70%, and 100% of the NYC Taxi dataset to evaluate performance and scalability in Python with pandas and Jupyter Notebook.

Filtering & Sorting with Sequential, Threading, and Multiprocessing
This project compares the performance of Sequential, Thread-based, and Multiprocessing approaches to perform filtering and sorting operations on the NYC Taxi Trip Duration dataset.

It evaluates how each method performs under different data scales: 25%, 50%, 70%, and 100% of the dataset.

Objectives
Filter rows where trip_duration > 1000

Sort rows based on trip_duration

Apply 3 processing strategies:

Sequential

Multithreaded

Multiprocessing

Compare performance at 4 data scales:

25%

50%

70%

100%

Dataset Sample (train.csv)
Column	Description
trip_duration	Duration of the trip in seconds
pickup_datetime	Pickup timestamp
dropoff_datetime	Dropoff timestamp
pickup_longitude	Pickup location longitude
dropoff_longitude	Dropoff location longitude
...	Other taxi trip features

Techniques Used
Method	Library	Parallel?	Output Type
Sequential	Base Python / Pandas	No	Filtered & sorted
Threading	threading	Yes (I/O-bound)	Filtered & sorted
Multiprocessing	multiprocessing	Yes (CPU-bound)	Filtered & sorted

Performance Measurement
Each approach is timed using Python’s time module. Results are printed after each run for direct comparison.

Example timing output:

matlab
Copy
Edit
Sequential (50%): 0.43 seconds
Threading   (50%): 0.31 seconds
Multiproc.  (50%): 0.19 seconds
Requirements
Python 3.7+

pandas

Jupyter Notebook

Install dependencies (if needed):

bash
Copy
Edit
pip install pandas
Usage
Open the main notebook:
filter_sort_comparison.ipynb

Run all cells.

Observe time differences and output data samples.

File Structure
bash
Copy
Edit
├── train.csv                      # Dataset (original or sampled)
├── filter_sort_comparison.ipynb  # Main notebook with all 3 approaches
├── README.md                     # Project description
Ideas for Extension
Export performance as charts (matplotlib / seaborn)

Use larger datasets for more realistic benchmarking

Add memory usage comparison

Try Dask or joblib for scalable solutions
