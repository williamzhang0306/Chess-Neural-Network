import numpy as np
import chess
from chess.engine import PlayResult
from keras import models
from strategies import ExampleEngine

class NeuralNetEngine2(ExampleEngine):

    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        print('NN Engine 2 created!')
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)

        try:
            self.model = models.load_model("/Users/williamzhang/Documents/College/chess project testing/Version 1/lichess-bot/Model_Cloud4_12_20_2")
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

