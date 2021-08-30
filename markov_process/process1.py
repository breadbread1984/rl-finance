#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;

def main(x0 = 1., step_time = 20, a1 = 0.25, L = 0):
  
  results = list();
  x = x0;
  results.append(x);
  for i in range(step_time):
    x = step(x, a1 = a1, L = L);
    results.append(x);
  return results;

def step(x_t, a1, L):

  up = tfp.distributions.Bernoulli(probs = tf.math.sigmoid(a1 * (L-x_t))).sample();
  x_tp1 = tf.where(tf.math.equal(up, 1), x_t + 1, x_t - 1);
  return x_tp1;

if __name__ == "__main__":
  print(main());

