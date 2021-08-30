#!/usr/bin/python3

from typing import TypeVar, Generic, Dict, List, Optional, Iterable;
from markov import MarkovProcess, Distribution, State;

# INFO: finite markov process can have finite states

class Transitions(Generic[State]):
  def __init__(self, transitions: Dict[State, Optional[Distribution[State]]]):
    self.transitions = transitions;
  def items(self,):
    return self.transitions.items();

class FiniteMarkovProcess(MarkovProcess[State]):
  def __init__(self, transitions: Transitions):
    self.non_terminal_state = [s for s, v in transitions.items() if v is not None];
    self.transitions = transitions;
  def transition(self, s: State) -> Optional[Distribution[State]]:
    return self.transitions[s];
  def states(self,) -> Iterable[State]:
    return self.transitions.keys();

