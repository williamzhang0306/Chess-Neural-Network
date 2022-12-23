import numpy as np
from keras import models

class ChessClassifier:

    def __init__(self):
        self.model = models.load_model("/Users/williamzhang/Documents/College/Neural-Network-Chess/keras-exports/Model_Cloud5_12_20_22")
        self.classes = {}

    def fen_to_image(self, fen_string: str) -> np.array:
        '''Converts fen string to 8x8x13 (rows x cols x channels) image. 
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
        active_color = parts[1]
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

        # add active side info
        # image[0,0,0] = 1 if active_color == 'w' else -1

        return image

    def classify_board(self, fen_string):
        network_input = np.empty([1,8,8,12])
        image = self.fen_to_image(fen_string)
        network_input[0,] = image

        prediction = self.model.predict(network_input)
        print(prediction)

