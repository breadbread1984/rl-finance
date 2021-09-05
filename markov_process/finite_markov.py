#!/usr/bin/python3

from typing import TypeVar, Generic, Dict, List, Optional, Iterable;
from dataclasses import dataclass;
from markov import MarkovProcess, Distribution, State;
import tensorflow as tf;
import tensorflow_probability as tfp;

# INFO: finite markov process can have finite states

class Transitions(Generic[State]):
  def __init__(self, transitions: Dict[State, Optional[Distribution[State]]]):
    self.transitions = transitions;
  def items(self,):
    return self.transitions.items();
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
          alpha_tp1 = state_t.position() - i;
          dist[InventoryState(alpha_tp1, beta_tp1)] = self.poisson.prob(i) if i < state_t.position() else self.poisson.cdf(i);
        transitions[state_t] = Distribution(dist);
    super(SimpleInventoryFMP, self).__init__(Transitions(transitions));
  def init_distribution(self, ) -> Distribution[InventoryState]:
    return Distribution({
      InventoryState(on_hand = 0, on_order = 0): 1.,
    });

if __name__ == "__main__":

  proc1 = SimpleInventoryFMP(capacity = 2, poisson_lambda = 1.);
  print(proc1.transitions);
  for state in proc1.simulate():
    print(state);

