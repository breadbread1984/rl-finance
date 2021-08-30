#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;

def main(x0 = 1., step_time = 20, a2 = 0.25):

  results = list();
  x_t = x0;
  results.append(x_t);
  x_tm1 = None;
  for i in range(step_time):
    x_tp1 = step(x_t, x_tm1, a2 = a2);
    x_tm1 = x_t;
    x_t = x_tp1;
    results.append(x_t);
  return results;

def step(x_t, x_tm1, a2):

  if x_tm1 is None:
    up = tfp.distributions.Bernoulli(probs = 0.5).sample();
  else:
    up = tfp.distributions.Bernoulli(probs = 0.5 * (1 - a2 * (x_t - x_tm1))).sample();
  x_tp1 = tf.where(tf.math.equal(up, 1), x_t + 1, x_t - 1);
  return x_tp1;

if __name__ == "__main__":
  print(main());

