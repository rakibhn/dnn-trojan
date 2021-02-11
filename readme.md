# HT-Trojan Detection using Deep Neural Network

# Data Description:

TrojanFree folder is aes_128 module without Trojan.

All the netlists and simulation results for each benchmark are stored in their folders.
In each folder, it contains the sdf(delay) file, netlist file, two csv files which data_0.000_00_1.csv is when #0.17 clk = ~clk;  data_0.000_00_2.csv is when #5 clk = ~clk;
The testbenchs for simulation is also added.
The first and second column are same for all the datasheet which is state and expected output, the third column is the real output and forth column is the additional output when the top module have other output.

How to Run:

1. Install all required dependency packages.
2. Run the HW_trojan_ml.py script (change the directory that directs to all the benchmark file {dir = "~/Data/"})