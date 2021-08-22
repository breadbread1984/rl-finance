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

def step(x0, a1, L):

  up = tfp.distributions.Bernoulli(probs = tf.math.sigmoid(-a1 * (x0-L))).sample();
  x1 = tf.where(tf.math.equal(up, 1), x0 + 1, x0 - 1);
  return x1;

if __name__ == "__main__":
  print(main());

