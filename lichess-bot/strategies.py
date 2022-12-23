"""
Some example strategies for people who want to create a custom, homemade bot.
And some handy classes to extend
"""
import numpy as np
from keras import models
import chess
from chess.engine import PlayResult
import random
from engine_wrapper import EngineWrapper

print('imports done')

model = models.load_model

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


class NeuralNetEngine(ExampleEngine):
    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)
        try:
            self.model = models.load_model("/Users/williamzhang/Documents/College/chess project testing/lichess-bot/Model_3_12_16_22")
            print("model loaded")
        except:
            print("MODEL FAILED TO LOAD")
            raise ValueError

    def board_to_NN_input(self,board):
        '''Creates a np array that can be used as input for the model from a chess.Board object'''
        result = np.zeros([1,7,8,8])
        for i in range(0,8):
            for j in range(0,8):
                square_index = 8*i + j
                square = chess.SQUARES[square_index]
                piece = board.piece_at(square)
                # there is no piece move on to the next square
                if piece == None: continue
                square_value = 1 if piece.color == chess.WHITE else -1
                piece_index = piece.piece_type
                result[0][piece_index][i][j] = square_value
        if board.turn == chess.WHITE:
            result[0][0][0][4] = 1
        else:
            result[0][0][7][4] = 1
        fen_info = board.fen().split(' ')
        castling_rights = fen_info[2]
        en_passant = fen_info[3]
        # got the next 12 lines of code from kaggle: user UTA-DCM0927
        castling = {
            'K': (7,7),
            'Q': (7,0),
            'k': (0,7),
            'q': (0,0),
        }
        if castling_rights != '-':
            for char in castling_rights:
                result[0, 0, castling[char][0], castling[char][1] ] = 1
        if en_passant != '-':
            result[0,0,int(en_passant[1]) - 1, ord(en_passant[0]) - ord('a')] = 1
        return result

    def search(self, board, *args):
        print('searching')
        
        moves = list(board.legal_moves)        
        # try:
        #     model = models.load_model("Model_2_12_16_22")
        #     print("model load succesfully")
        # except:
        #     model = None
        #     print('model failed to load hm')
        evaluations = []

        for move in moves:
            board.push(move)
            try:
                model_input = self.board_to_NN_input(board)
            except:
                model_input = np.zeros([1,7,8,8])
            evaluations.append(self.model.predict(model_input)[0][0])
            board.pop()


        if board.turn == chess.WHITE:
            tmp = max(evaluations)
        else:
            tmp = min(evaluations)
            
        index = evaluations.index(tmp)
        best_move = moves[index]
        print(best_move,tmp)
        return PlayResult(best_move, None)

class NeuralNetEngine2(ExampleEngine):

    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        print('NN Engine 2 created!')
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)

        try:
            self.model = models.load_model("/Users/williamzhang/Documents/College/chess project testing/lichess-bot/Model_Cloud5_12_20_22")
            print("Cloud Trained Model loaded")
        except:
            print("MODEL FAILED TO LOAD")
            raise ValueError

    def fen_to_image(self, fen_string: str) -> np.array:
        '''Converts fen string to 8x8x12 (rows x cols x channels) image. 
        13 channels represent 12 pieces plue one channel for white or black is playing.
        In each pixel of the image, 1 represents a white piece is present, 
        -1 black piece present, 0 no piece present.'''
        # used for conversion
        piece_to_channel = {'P':0, 'N':1, 'B':2, 'R':3 , 'Q':4 , 'K':5,
                          'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
        image = np.zeros([8,8,12])

        # extract info from the fen string
        parts = fen_string.split(" ")
        fen_board = parts[0].split('/')
        #active_color = parts[1]
        # populate the board image
        for i, row in enumerate(fen_board):
            j = 0
            for char in row:
                # if the character a piece, e.g. 'p' or 'K'
                if char in piece_to_channel:
                    # 1 if the piece is white, -1 if the piece is black
                    val = 1 if char.isupper() else -1
                    image[i, j, piece_to_channel[char]] = val
                    j += 1

                # otherwise the character is a number (number of empty squares in the row)
                else:
                    j += int(char)

        out = np.empty([1,8,8,12])
        out[0,] = image

        return out

    def search(self, board, *args):
        print('searching')
        
        moves = list(board.legal_moves)        
        evaluations = []

        for move in moves:
            board.push(move)
            try:
                model_input = self.fen_to_image(board.fen())
                print('input conversion succeded')
            except:
                model_input = np.zeros([1,8,8,12])
                print('input conversion failed')
            evaluations.append(self.model.predict(model_input)[0][0])
            board.pop()


        if board.turn == chess.WHITE:
            tmp = max(evaluations)
        else:
            tmp = min(evaluations)
            
        index = evaluations.index(tmp)
        best_move = moves[index]
        print(best_move,tmp)
        return PlayResult(best_move, None)

class NNEngine_with_Depth(NeuralNetEngine2):

    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)

    def serach(self, board, *args):
        raise NotImplementedError