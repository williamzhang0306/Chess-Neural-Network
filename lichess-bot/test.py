import numpy as np
import chess
from keras import models

def fen_to_trit_array(fen_string:str ) -> np.array:
    '''Converts a fen string to an 7x8x8 tensor. I think of them as 7 8x8 boards. 
    6 boards are used to store locations for the 6 piece types. 1 for white piece, 0 for no piece -1 for black piece at a square.
    7th board stores game information like castling rights, enpassant squares, and active side'''

    trit_array = np.zeros([7,8,8])
    board = chess.Board(fen_string)

    # put pieces in boards 1 through 6
    for i in range(0,8):
        for j in range(0,8):
            square_index = 8*i + j
            square = chess.SQUARES[square_index]
            piece = board.piece_at(square)

            # there is no piece move on to the next square
            if piece == None: continue

            # the piece is white (white == True :/)    
            trit_value = 1 if piece.color else -1
            piece_index = piece.piece_type
            trit_array[piece_index][i][j] = trit_value

    # put game info in board 0
    active_color = 'white' if board.turn else 'black'
    if active_color == 'white':
        trit_array[0][0][4] = 1
    else:
        trit_array[0][7][4] = 1

    fen_info = fen_string.split(" ")
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
            trit_array[0, castling[char][0], castling[char][1]] = 1
    if en_passant != '-':
        trit_array[0,int(en_passant[1]) - 1, ord(en_passant[0]) - ord('a')] = 1

    return trit_array

example_input = np.zeros([1,7,8,8])


model = models.load_model("Model_2_12_16_22")

print(model.predict(example_input))
print(model.summary())