#!/usr/bin/python3

from typing import TypeVar, Generic, Dict, List, Optional, Iterable;
from dataclasses import dataclass;
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
from markov import MarkovProcess, Distribution, State;

# INFO: finite markov process can have finite states

class Transitions(Generic[State]):
  def __init__(self, transitions: Dict[State, Optional[Distribution[State]]]):
    self.transitions = transitions;
  def items(self,):
    return self.transitions.items();
  def __getitem__(self, key):
    return self.transitions[key];
  def __str__(self,):
    msg: str = '';
    for key, value in self.transitions.items():
      msg += "From State %s:\n%s" % (str(key), value if value else '');
    return msg;

class FiniteMarkovProcess(MarkovProcess[State]):
  def __init__(self, transitions: Transitions):
    self.non_terminal_state = [s for s, v in transitions.items() if v is not None];
    self.transitions = transitions;
  def transition(self, s: State) -> Optional[Distribution[State]]:
    return self.transitions[s];
  def states(self,) -> Iterable[State]:
    return self.transitions.keys();
  def get_transition_matrix(self,) -> np.ndarray:
    mat = np.zeros((len(self.non_terminal_state), len(self.non_terminal_state)));
    for i, s1 in enumerate(self.non_terminal_state):
      for j, s2 in enumerate(self.non_terminal_state):
        mat[i,j] = self.transitions[s1][s2];
    return mat;
  def get_stationary_distribution(self,) -> Distribution[State]:
    vals, vecs = np.linalg.eig(np.transpose(self.get_transition_matrix()));
    val = np.real(vecs[:,np.where(np.abs(vals - 1) < 1e-7)[0][0]]);
    return Distribution({self.non_terminal_state[i]: v for i, v in enumerate(val / sum(val))});

@dataclass
class InventoryState:
  on_hand: int;
  on_order: int;
  def __hash__(self,):
    return int(str(self.on_hand) + str(self.on_order));
  def position(self,):
    return self.on_hand + self.on_order;

class SimpleInventoryFMP(FiniteMarkovProcess[InventoryState]):
  def __init__(self, capacity: int, poisson_lambda: float):
    self.poisson = tfp.distributions.Poisson(rate = poisson_lambda);
    # INFO: creating transition matrix
    # because 0 <= on_hand + on_order <= capacity
    # and 0 <= on_hand <= capacity
    # so 0 <= on_order <= capacity - on_hand
    transitions: Dict[InventoryState, Optional[Distribution[InventoryState]]] = dict();
    for alpha in range(capacity + 1):
      for beta in range(capacity + 1 - alpha):
        state_t = InventoryState(alpha, beta);
        beta_tp1 = capacity - state_t.position(); # ordering policy given in page 52
        dist = dict();
        # iterate over all possible number of sold
        for i in range(state_t.position() + 1):
          alpha_tp1 = state_t.position() - i; # position - sold
          # NOTE: define p(sold = position) = sum from {i = position} to {+inf} {poisson.prob(sold)} = 1 - poisson.cdf(position - 1)
          dist[InventoryState(alpha_tp1, beta_tp1)] = self.poisson.prob(i) if i < state_t.position() else 1 - self.poisson.cdf(state_t.position() - 1);
        transitions[state_t] = Distribution(dist);
    super(SimpleInventoryFMP, self).__init__(Transitions(transitions));
  def init_distribution(self, ) -> Distribution[InventoryState]:
    return Distribution({
      InventoryState(on_hand = 0, on_order = 0): 1.,
    });

if __name__ == "__main__":

  proc1 = SimpleInventoryFMP(capacity = 2, poisson_lambda = 1.);
  print(proc1.transitions);
  sim_iter = proc1.simulate();
  for i in range(10):
    state = next(sim_iter);
    print(state);
  m = proc1.get_transition_matrix();
  print(m);
  print(proc1.get_stationary_distribution());
