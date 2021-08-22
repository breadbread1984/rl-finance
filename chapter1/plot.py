#!/usr/bin/python3

import matplotlib.pyplot as plt;
from process1 import main as proc1;
from process2 import main as proc2;
from process3 import main as proc3;

def main():
  time = [t for t in range(101)];
  results1 = proc1(step_time = 100, x0 = 100., a1 = 0.25, L = 100.);
  results2 = proc2(step_time = 100, x0 = 100., a2 = 0.75);
  results3 = proc3(step_time = 100, x0 = 100., a3 = 1.);
  plt.plot(time, results1, label = 'process1', color = 'red');
  plt.plot(time, results2, label = 'process2', color = 'green');
  plt.plot(time, results3, label = 'process3', color = 'blue');
  plt.title('stock price');
  plt.xlabel('time');
  plt.ylabel('price');
  plt.legend();
  plt.show();

if __name__ == "__main__":

  main();

