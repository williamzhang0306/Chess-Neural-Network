"""
Some example strategies for people who want to create a custom, homemade bot.
And some handy classes to extend
"""
import os
import sys
from chess.engine import PlayResult
import random
from engine_wrapper import EngineWrapper

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EngineClasses import *

class FillerEngine:
    """
    Not meant to be an actual engine.

    This is only used to provide the property "self.engine"
    in "MinimalEngine" which extends "EngineWrapper"
    """
    def __init__(self, main_engine, name=None):
        self.id = {
            "name": name
        }
        self.name = name
        self.main_engine = main_engine

    def __getattr__(self, method_name):
        main_engine = self.main_engine

        def method(*args, **kwargs):
            nonlocal main_engine
            nonlocal method_name
            return main_engine.notify(method_name, *args, **kwargs)

        return method


class MinimalEngine(EngineWrapper):
    """
    Subclass this to prevent a few random errors

    Even though MinimalEngine extends EngineWrapper,
    you don't have to actually wrap an engine.

    At minimum, just implement `search`,
    however you can also change other methods like
    `notify`, `first_search`, `get_time_control`, etc.
    """
    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(options, draw_or_resign)

        self.engine_name = self.__class__.__name__ if name is None else name

        self.engine = FillerEngine(self, name=self.name)
        self.engine.id = {
            "name": self.engine_name
        }

    def search(self, board, time_limit, ponder, draw_offered, root_moves):
        """
        The method to be implemented in your homemade engine

        NOTE: This method must return an instance of "chess.engine.PlayResult"
        """
        raise NotImplementedError("The search method is not implemented")

    def notify(self, method_name, *args, **kwargs):
        """
        The EngineWrapper class sometimes calls methods on "self.engine".
        "self.engine" is a filler property that notifies <self>
        whenever an attribute is called.

        Nothing happens unless the main engine does something.

        Simply put, the following code is equivalent
        self.engine.<method_name>(<*args>, <**kwargs>)
        self.notify(<method_name>, <*args>, <**kwargs>)
        """
        pass


class ExampleEngine(MinimalEngine):
    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    def search(self, board, *args):
        print('ha')
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Gets the first move when sorted by uci representation"""
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)

# my engines

class NegaMaxEngine(MinimalEngine):
    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)
        self.engine = SimpleChessEngine(board = None, depth = 1)

    def search(self, board, time_limit, ponder, draw_offered, root_moves):
        self.engine.board = board
        result = self.engine.get_move()
        move = result[0]
        eval = result[1]
        num_searches = result[2]

        print(move,eval,num_searches)

        return PlayResult(move,None)


class AiNegaMax(NegaMaxEngine):
    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)
        model_path = "../../keras-exports/Model_Cloud13_01_01_23"
        self.engine = TFLiteEngine(model=model_path,board = None, depth = 3)