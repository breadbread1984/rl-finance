#!/usr/bin/python3

import numpy as np;
import matplotlib.pyplot as plt;
from process1 import main as proc1;
from process2 import main as proc2;
from process3 import main as proc3;

def main():
  # 1) simulate stock prices
  time = [t for t in range(101)];
  results1 = proc1(step_time = 100, x0 = 100., a1 = 0.25, L = 100.);
  results2 = proc2(step_time = 100, x0 = 100., a2 = 0.75);
  results3 = proc3(step_time = 100, x0 = 100., a3 = 1.);
  # 2) plot stock prices
  plt.figure();
  plt.plot(time, results1, label = 'process1', color = 'red');
  plt.plot(time, results2, label = 'process2', color = 'green');
  plt.plot(time, results3, label = 'process3', color = 'blue');
  plt.title('stock price');
  plt.xlabel('time');
  plt.ylabel('price');
  plt.legend();
  # 3) plot price histogram
  total_results1 = np.zeros((0,), dtype = np.float32);
  total_results2 = np.zeros((0,), dtype = np.float32);
  total_results3 = np.zeros((0,), dtype = np.float32);
  for i in range(1000):
    results1 = proc1(step_time = 100, x0 = 100., a1 = 0.25, L = 100.);
    results2 = proc2(step_time = 100, x0 = 100., a2 = 0.75);
    results3 = proc3(step_time = 100, x0 = 100., a3 = 1.);
    total_results1 = np.concatenate([total_results1, results1], axis = 0);
    total_results2 = np.concatenate([total_results2, results2], axis = 0);
    total_results3 = np.concatenate([total_results3, results3], axis = 0);
  plt.figure();
  hist1 = np.unique(total_results1, return_counts = True);
  hist2 = np.unique(total_results2, return_counts = True);
  hist3 = np.unique(total_results3, return_counts = True);
  plt.plot(hist1[0], hist1[1], label = 'process1', color = 'red');
  plt.plot(hist2[0], hist2[1], label = 'process2', color = 'green');
  plt.plot(hist3[0], hist3[1], label = 'process3', color = 'blue');
  plt.title('stock price histogram');
  plt.xlabel('prices');
  plt.ylabel('count');
  plt.legend();
  
  plt.show();

if __name__ == "__main__":

  main();

