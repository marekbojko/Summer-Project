# -*- coding: utf-8 -*-
"""
Prisoner's Dilemma

Adaptation of bits from the axelrod library.
"""

from enum import Enum
from functools import total_ordering
from typing import Iterable, Tuple, Union
import copy
import inspect
import itertools
import types
from typing import Any, Dict
from collections import Counter
import warnings
import random
import numpy as np
import math

np.random.seed(None)


def toroidal_distance (pos_1: Tuple[float,float], pos_2: Tuple[float,float]) -> float:
    dx = abs(pos_1[0]-pos_2[0])
    dy = abs(pos_1[1]-pos_2[1])
    """
    if dx > 0.5:
        dx = 1 - dx
    if dy > 0.5:
        dy = 1 - dy
    """
    return math.sqrt(dx**2 + dy**2)


class Action(Enum):
    """Core actions in the Prisoner's Dilemma.
    There are only two possible actions, namely Cooperate (C) or Defect (D) - 
    in our case cooperate = share information and defect = notC
    """

    C = 1  # Cooperate
    D = 0  # Defect

    def __lt__(self, other):
        return self.value < other.value

    def flip(self):
        """Returns the opposite Action."""
        if self == Action.C:
            return Action.D
        return Action.C

    @classmethod
    def from_char(cls, character):
        """Converts a single character into an Action.
        """
        if character == "C":
            return cls.C
        if character == "D":
            return cls.D
        raise UnknownActionError('Character must be "C" or "D".')


def str_to_actions(actions: str) -> tuple:
    """Converts a string to a tuple of actions.
    """
    return tuple(Action.from_char(element) for element in actions)


def actions_to_str(actions: Iterable[Action]) -> str:
    """Converts an iterable of actions into a string.
    """
    return "".join(map(str, actions))
    

C, D = Action.C, Action.D
    

class Game(object):
    """Container for the game matrix and scoring logic.
    Attributes
    ----------
    scores: dict
        The numerical score attribute to all combinations of action pairs.
    """

    def __init__(self, r= 1, s= -1, t= 1, p= 0) -> None:
        self.scores = {(C, C): (r, r), (D, D): (p, p), (C, D): (s, t), (D, C): (t, s)}

    def RPST(self):
        """Press and Dyson notation"""
        R = self.scores[(C, C)][0]
        P = self.scores[(D, D)][0]
        S = self.scores[(C, D)][0]
        T = self.scores[(D, C)][0]
        return R, P, S, T

    def score(self, pair: Tuple[Action, Action]):
        """Returns the appropriate score for a decision pair."""
        return self.scores[pair]

    def __eq__(self, other):
        if not isinstance(other, Game):
            return False
        return self.RPST() == other.RPST()


DefaultGame = Game()


class History(object):
    """
    History class to track the history of play and metadata including
    the number of cooperations and defections, and if available, the
    opponents plays and the state distribution of the history of play.
    """

    def __init__(self, plays=None, coplays=None):
        self._plays = []
        # Coplays is tracked mainly for computation of the state distribution
        # when cloning or dualing.
        self._coplays = []
        self._actions = Counter()
        self._state_distribution = Counter()
        if plays:
            self.extend(plays, coplays)

    def append(self, play, coplay):
        """Appends a new (play, coplay) pair an updates metadata for
        number of cooperations and defections, and the state distribution."""
        self._plays.append(play)
        self._actions[play] += 1
        self._coplays.append(coplay)
        self._state_distribution[(play, coplay)] += 1

    def copy(self):
        """Returns a new object with the same data."""
        return self.__class__(plays=self._plays, coplays=self._coplays)

    def extend(self, plays, coplays):
        """A function that emulates list.extend."""
        # We could repeatedly call self.append but this is more efficient.
        self._plays.extend(plays)
        self._actions.update(plays)
        self._coplays.extend(coplays)
        self._state_distribution.update(zip(plays, coplays))

    def reset(self):
        """Clears all data in the History object."""
        self._plays.clear()
        self._coplays.clear()
        self._actions.clear()
        self._state_distribution.clear()

    @property
    def coplays(self):
        return self._coplays

    @property
    def cooperations(self):
        return self._actions[C]

    @property
    def defections(self):
        return self._actions[D]

    @property
    def state_distribution(self):
        return self._state_distribution

    def __eq__(self, other):
        if isinstance(other, list):
            return self._plays == other
        elif isinstance(other, History):
            return self._plays == other._plays and self._coplays == other._coplays
        raise TypeError("Cannot compare types.")

    def __getitem__(self, key):
        # Passthrough keys and slice objects
        return self._plays[key]

    def __str__(self):
        return actions_to_str(self._plays)

    def __list__(self):
        return self._plays

    def __len__(self):
        return len(self._plays)

    def __repr__(self):
        return repr(self.__list__())
    

class history_all(object):
    def __init__(self,coplayers: list=[]):
        self.histories = self.initialise_dict(coplayers)
        self.coplayers = coplayers
        self.distribution = self.get_dist_all_interactions()
        
    def initialise_dict(self,coplayers: list=[]):
        histories = dict()
        #print (coplayers)
        for c in coplayers:
            histories[c] = History()
        #print (histories)
        return histories
    
    def get_dist_all_interactions(self):
        distribution = Counter({})
        for c in self.coplayers:
            distribution += self.histories[c].state_distribution
        return distribution
    

def compute_cooperations(interactions):
    """Returns the count of cooperations by each player for a set of
    interactions"""

    if len(interactions) == 0:
        return None

    cooperation = tuple(
        sum([play[player_index] == C for play in interactions])
        for player_index in [0, 1]
    )
    return cooperation


def compute_normalised_cooperation(interactions):
    """Returns the count of cooperations by each player per turn for a set of
    interactions"""
    if len(interactions) == 0:
        return None

    num_turns = len(interactions)
    cooperation = compute_cooperations(interactions)

    normalised_cooperation = tuple([c / num_turns for c in cooperation])

    return normalised_cooperation
    

def simultaneous_play(player, coplayer, noise=0):
    """This pits two players against each other."""
    s1, s2 = player.strategy(coplayer, coplayer.player_index), coplayer.strategy(player, player.player_index)
    if noise:
        s1 = random_flip(s1, noise)
        s2 = random_flip(s2, noise)
    player.update_history(s1, s2, coplayer.player_index)
    coplayer.update_history(s2, s1, player.player_index)
    return s1, s2


class Player(object):
    """A class for a player in the tournament.
    This is an abstract base class, not intended to be used directly.
    """

    name = "Player"
    classifier = {}  # type: Dict[str, Any]
    default_classifier = {
        "stochastic": False,
        "memory_depth": float("inf"),
        "makes_use_of": None,
        "long_run_time": False,
        "inspects_source": None,
        "manipulates_source": None,
        "manipulates_state": None,
    }

    # def __new__(cls, *args, history=None, **kwargs):
    def __new__(cls, *args, **kwargs):
        """Caches arguments for Player cloning."""
        obj = super().__new__(cls)
        obj.init_kwargs = cls.init_params(*args, **kwargs)
        return obj

    @classmethod
    def init_params(cls, *args, **kwargs):
        """
        Return a dictionary containing the init parameters of a strategy
        (without 'self').
        Use *args and *kwargs as value if specified
        and complete the rest with the default values.
        """
        sig = inspect.signature(cls.__init__)
        # The 'self' parameter needs to be removed or the first *args will be
        # assigned to it
        self_param = sig.parameters.get("self")
        new_params = list(sig.parameters.values())
        new_params.remove(self_param)
        sig = sig.replace(parameters=new_params)
        boundargs = sig.bind_partial(*args, **kwargs)
        boundargs.apply_defaults()
        return boundargs.arguments

    def __init__(self):
        """Initiates an empty history."""
        self._history = history_all()
        self.classifier = copy.deepcopy(self.classifier)
        for dimension in self.default_classifier:
            if dimension not in self.classifier:
                self.classifier[dimension] = self.default_classifier[dimension]
        #self.set_match_attributes()

    
    def set_match_attributes(self, length=-1, game=None, noise=0):
        if not game:
            game = DefaultGame
        self.match_attributes = {"length": length, "game": game, "noise": noise}
        self.receive_match_attributes()
    
    
    def __repr__(self):
        """The string method for the strategy.
        Appends the `__init__` parameters to the strategy's name."""
        name = self.name
        prefix = ": "
        gen = (value for value in self.init_kwargs.values() if value is not None)
        for value in gen:
            try:
                if issubclass(value, Player):
                    value = value.name
            except TypeError:
                pass
            name = "".join([name, prefix, str(value)])
            prefix = ", "
        return name

    def __getstate__(self):
        """Used for pickling. Override if Player contains unpickleable attributes."""
        return self.__dict__

    def strategy(self, opponent):
        """This is a placeholder strategy."""
        raise NotImplementedError()

    def play(self, opponent, noise=0):
        """This pits two players against each other."""
        return simultaneous_play(self, opponent, noise)

    def clone(self):
        """Clones the player without history, reapplying configuration
        parameters as necessary."""

        # You may be tempted to re-implement using the `copy` module
        # Note that this would require a deepcopy in some cases and there may
        # be significant changes required throughout the library.
        # Consider overriding in special cases only if necessary
        cls = self.__class__
        new_player = cls(**self.init_kwargs)
        new_player.match_attributes = copy.copy(self.match_attributes)
        return new_player

    def reset(self):
        """Resets a player to its initial state
        This method is called at the beginning of each match (between a pair
        of players) to reset a player's state to its initial starting point.
        It ensures that no 'memory' of previous matches is carried forward.
        """
        # This also resets the history.
        self.__init__(**self.init_kwargs)

    def update_history(self, play, coplay, opponent_index: int):
        self.history.histories[opponent_index].append(play, coplay)

    @property
    def history(self):
        return self._history

    # Properties maintained for legacy API, can refactor to self.history.X
    # in 5.0.0 to reduce function call overhead.
    
    def cooperations(self,opponent_index):
        return self._history.histories[opponent_index].cooperations

    
    def defections(self,opponent_index):
        return self._history.histories[opponent_index].defections

    
    def state_distribution(self,opponent_index):
        return self._history.histories[opponent_index].state_distribution
    

def random_choice(p: float = 0.5) -> Action:
    if p == 0:
        return D
    if p == 1:
        return C
    r = random.random()
    #print (r)
    if r < p:
        return C
    return D

    
class MemoryOnePlayer(Player):
    """
    Uses a four-vector for strategies based on the last round of play,
    (P(C|CC), P(C|CD), P(C|DC), P(C|DD)). 
    Memory One: see Nowak(1990)
    """

    def __init__(self, four_vector: Tuple[float, float, float, float] = None, initial_prob: float=0.5,player_index=0,pos: Tuple[float,float]=(0,0),coplayers: list=[]):
        """
        Parameters:
        fourvector: list or tuple of floats of length 4
            The response probabilities to the preceding round of play
            ( P(C|CC), P(C|CD), P(C|DC), P(C|DD) )
        initial: C or D
        
        Special Cases:
        Alternator is equivalent to MemoryOnePlayer((0, 0, 1, 1), C)
        Cooperator is equivalent to MemoryOnePlayer((1, 1, 1, 1), C)
        Defector   is equivalent to MemoryOnePlayer((0, 0, 0, 0), C)
        Random     is equivalent to MemoryOnePlayer((0.5, 0.5, 0.5, 0.5))
        (with a random choice for the initial state)
        TitForTat  is equivalent to MemoryOnePlayer((1, 0, 1, 0), C)
        WinStayLoseShift is equivalent to MemoryOnePlayer((1, 0, 0, 1), C)
        """
        super().__init__()
        self._initial = random_choice(initial_prob)
        self.initial_prob = initial_prob
        self.set_initial_four_vector(four_vector)
        self._history = history_all(coplayers)
        self.player_index = player_index
        self.loc = pos

    def set_initial_four_vector(self, four_vector):
        if four_vector is None:
            four_vector = (1, 0, 0, 1)
            warnings.warn("Memory one player is set to default (1, 0, 0, 1).")

        self.set_four_vector(four_vector)
        if self.name == "Generic Memory One Player":
            self.name = "%s: %s" % (self.name, four_vector)

    def set_four_vector(self, four_vector: Tuple[float, float, float, float]):
        if not all(0 <= p <= 1 for p in four_vector):
            raise ValueError(
                "An element in the probability vector, {}, is not "
                "between 0 and 1.".format(str(four_vector))
            )

        self._four_vector = dict(zip([(C, C), (C, D), (D, C), (D, D)], four_vector))
        self.classifier["stochastic"] = any(0 < x < 1 for x in set(four_vector))

    def strategy(self, opponent: Player, opponent_index: int) -> Action:
        if len(opponent.history.histories[self.player_index]) == 0:
            return random_choice(self.initial_prob*math.exp(toroidal_distance(self.loc,opponent.loc)*math.log(0.05)/math.sqrt(2)))
        # Determine which probability to use
        p = self._four_vector[(self.history.histories[opponent_index][-1], opponent.history.histories[self.player_index][-1])]
        p_share = p*math.exp(toroidal_distance(self.loc,opponent.loc)*math.log(0.05)/math.sqrt(2))
        #print (p)
        # Draw a random number in [0, 1] to decide
        return random_choice(p_share)

        
        
class ReactivePlayer(MemoryOnePlayer):
    """
    A generic reactive player. Defined by 2 probabilities conditional on the
    opponent's last move: P(C|C), P(C|D).
    Names:
    - Reactive: [Nowak1989]_
    """

    name = "Reactive Player"

    def __init__(self, probabilities: Tuple[float, float], initial_prob: float=0.5,player_index=0,coplayers: list=[],pos: Tuple[float,float]=(0,0)) -> None:
        four_vector = (*probabilities, *probabilities)
        super().__init__(four_vector)
        self.prob = probabilities
        self.initial_prob = initial_prob
        self._history = history_all(coplayers)
        self._initial = random_choice(initial_prob)
        self.player_index = player_index
        self.loc = pos
        self.name = "%s: %s" % (self.name, probabilities)
        
        
"""

P1 = ReactivePlayer((0.5,0.4),0.5,1,[2,3])
P2 = ReactivePlayer((0.8,0.4),0.5,2,[1,3])
P3 = ReactivePlayer((0.7,0.3),0.5,3,[1,2])
for t in range(5):
    simultaneous_play(P1, P2)
    simultaneous_play(P2, P3)
print (P1.history.histories[2])
print (P1.history.histories[2].state_distribution)

print (exp(0.5*math.log(0.15)/math.sqrt(2)))

"""