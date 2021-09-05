#!/usr/bin/python3

from typing import TypeVar, Generic, Dict, List, Optional, Iterable;
from abc import ABC, abstractmethod;
from dataclasses import dataclass;
import tensorflow as tf;
import tensorflow_probability as tfp;

# INFO: markov process can have inifinite countable states

# template variable
State = TypeVar('State');

# template class distribution
class Distribution(Generic[State]):
  def __init__(self, options: Dict[State, float]):
    self.keys: List[State] = list(options.keys());
    probs: List[float] = list(options.values());
    self.dist = tfp.distributions.Categorical(probs = probs);
  def sample(self,) -> State:
    s = self.dist.sample();
    return self.keys[s];
  def __str__(self,) -> str:
    msg: str = '';
    for i in range(len(self.keys)):
      msg += "\tTo State %s with Probability %f\n" % (str(self.keys[i]), self.dist.prob(i));
    return msg;

# template abstract class markov process
class MarkovProcess(ABC, Generic[State]):
  @abstractmethod
  def transition(self, state: State) -> Optional[Distribution[State]]:
    pass;
  def is_terminal(self, state: State) -> bool:
    return self.transition(state) is None;
  @abstractmethod
  def init_distribution(self,) -> Distribution[State]:
    pass;
  def simulate(self,) -> Iterable[State]:
    state: State = self.init_distribution().sample();
    while True:
      yield state;
      next_state_dist = self.transition(state);
      if next_state_dist is None: return;
      state = next_state_dist.sample();

# value to template variable
@dataclass
class StateMP1:
  up_count: int;
  down_count: int;
  def __hash__(self,):
    return int(str(self.up_count) + str(self.down_count));

class Process1(MarkovProcess[StateMP1]):
  def __init__(self, x0: float = 100., alpha1: float = 0.25, L: float = 100., step_time: int = 0):
    assert step_time >= 0;
    self.x0: float = x0;
    self.alpha1: float = alpha1;
    self.L: float = L;
    self.step_time = step_time;
  def transition(self, state: StateMP1) -> Optional[Distribution[StateMP1]]:
    x_t = self.x0 + state.up_count - state.down_count;
    up_prob = tf.math.sigmoid(self.alpha1 * (self.L - x_t));
    if self.step_time == 0 or state.up_count + state.down_count < self.step_time:
      return Distribution({
        StateMP1(up_count = state.up_count + 1, down_count = state.down_count): up_prob,
        StateMP1(up_count = state.up_count, down_count = state.down_count + 1): 1 - up_prob,
      });
    else:
      return None;
  def init_distribution(self,) -> Distribution[StateMP1]:
    return Distribution({
      StateMP1(up_count = 0, down_count = 0): 1.,
    });

@dataclass
class StateMP2:
  up_count_t: int;
  down_count_t: int;
  up_count_tm1: int;
  down_count_tm1: int;
  def __hash__(self,):
    return int(str(self.up_count_t) + str(self.down_count_t) + str(self.up_count_tm1) + str(self.down_count_tm1));

class Process2(MarkovProcess[StateMP2]):
  def __init__(self, x0: float = 100., alpha2: float = 0.25, step_time: int = 0):
    assert step_time >= 0;
    self.x0: float = x0;
    self.alpha2: float = alpha2;
    self.step_time = step_time;
  def transition(self, state: StateMP2) -> Optional[Distribution[StateMP2]]:
    if self.step_time == 0 or state.up_count_t + state.down_count_t < self.step_time:
      if state.up_count_t == 0 and state.down_count_t == 0:
        return Distribution({
          StateMP2(up_count_tm1 = state.up_count_t, down_count_tm1 = state.down_count_t, up_count_t = state.up_count_t + 1, down_count_t = state.down_count_t): 0.5,
          StateMP2(up_count_tm1 = state.up_count_t, down_count_tm1 = state.down_count_t, up_count_t = state.up_count_t, down_count_t = state.down_count_t + 1): 0.5,
        });
      else:
        x_t = self.x0 + state.up_count_t - state.down_count_t;
        x_tm1 = self.x0 + state.up_count_tm1 - state.down_count_tm1;
        up_prob = 0.5 * (1 - self.alpha2 * (x_t - x_tm1));
        return Distribution({
          StateMP2(up_count_tm1 = state.up_count_t, down_count_tm1 = state.down_count_t, up_count_t = state.up_count_t + 1, down_count_t = state.down_count_t): up_prob,
          StateMP2(up_count_tm1 = state.up_count_t, down_count_tm1 = state.down_count_t, up_count_t = state.up_count_t, down_count_t = state.down_count_t + 1): 1 - up_prob,
        });
    else:
      return None;
  def init_distribution(self,) -> Distribution[StateMP2]:
    return Distribution({
      StateMP2(up_count_tm1 = 0, down_count_tm1 = 0, up_count_t = 0, down_count_t = 0): 1.
    });

@dataclass
class StateMP3:
  up_count: int;
  down_count: int;
  def __hash__(self,):
    return int(str(self.up_count) + str(self.down_count));

class Process3(MarkovProcess[StateMP3]):
  def __init__(self, x0: float = 100., alpha3: float = 0.25, step_time: int = 0):
    assert step_time >= 0;
    self.x0: float = x0;
    self.alpha3: float = alpha3;
    self.step_time = step_time;
  def transition(self, state: StateMP3) -> Optional[Distribution[StateMP3]]:
    if self.step_time == 0 or state.up_count + state.down_count < self.step_time:
      if state.up_count == 0 and state.down_count == 0:
        return Distribution({
          StateMP3(up_count = state.up_count + 1, down_count = state.down_count): 0.5,
          StateMP3(up_count = state.up_count, down_count = state.down_count + 1): 0.5,
        });
      else:
        up_prob = 1 / (1 + (state.up_count/state.down_count)**self.alpha3);
        return Distribution({
          StateMP3(up_count = state.up_count + 1, down_count = state.down_count): up_prob,
          StateMP3(up_count = state.up_count, down_count = state.down_count + 1): 1 - up_prob,
        });
    else:
      return None;
  def init_distribution(self,) -> Distribution[StateMP3]:
    return Distribution({
      StateMP3(up_count = 0, down_count = 0): 1.
    });

if __name__ == "__main__":
  proc1 = Process1(step_time = 10);
  for state in proc1.simulate():
    print(state);
  proc2 = Process2(step_time = 10);
  for state in proc2.simulate():
    print(state);
  proc3 = Process3(step_time = 10);
  for state in proc3.simulate():
    print(state);
