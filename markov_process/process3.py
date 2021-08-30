#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;

def main(x0 = 1., step_time = 20, a3 = 0.25):

  results = list();
  x_t = x0;
  results.append(x_t);
  for i in range(step_time):
    x_t = step(results, a3);
    results.append(x_t);
  return results;

def step(history, a3):

  history = np.array(history);
  assert tf.shape(history)[0] > 0;
  if tf.shape(history)[0] == 1:
    up = tfp.distributions.Bernoulli(probs = 0.5).sample();
  else:
    diff = history[1:] - history[:-1];
    u = tf.where(tf.math.greater(diff, 0),diff,tf.zeros_like(diff));
    d = tf.where(tf.math.less(diff,0),-diff,tf.zeros_like(diff));
    U = tf.math.reduce_sum(u);
    D = tf.math.reduce_sum(d);
    up = tfp.distributions.Bernoulli(probs = 1/(1+(U/D)**a3)).sample();
  x_tp1 = tf.where(tf.math.equal(up, 1), history[-1] + 1, history[-1] - 1);
  return x_tp1;

if __name__ == "__main__":
  print(main());

