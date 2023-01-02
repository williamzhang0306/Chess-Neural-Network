# .csv and .db files in the ../data are too large to load into memory by themselves, so I had to make a data generator to only load data that can fit in memory
# code for the generator is based on a tutorial written by Afshine and Shervine Amidi.
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import pandas as pd
import keras
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_frame: pd.DataFrame, IDs: list, batch_size = 32, x_dim = (8,8,12), y_dim = (1,1), shuffle = True):
        
        self.dataframe = data_frame
        self.IDs = IDs
        self.batch_size = batch_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

        # self.IDs correspond to the identifiers within the data frame to locate the FEN and Stockfish Evaluation
        # self.indexes are the indexes within the tensor object that is returned
        # there are as many indexes as IDs
        # e.g. Ids = ['a','d','b','z']
        # e.g. indexes = [ 0,  1,  2,  3]

    def on_epoch_end(self):
        'Update indexes at the end of an epoch'
        self.indexes = np.arange(len(self.IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def fen_to_image(self, fen_string: str) -> np.array:
        '''Converts fen string to 8x8x13 (rows x cols x channels) image. 
        13 channels represent 12 pieces plue one channel for white or black is playing.
        In each pixel of the image, 1 represents a white piece is present, 
        -1 black piece present, 0 no piece present.'''

        # used for conversion
        piece_to_channel = {'P':0, 'N':1, 'B':2, 'R':3 , 'Q':4 , 'K':5,
                          'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
        
        # initialize empty image
        image = np.zeros(self.x_dim)

        # extract info from the fen string
        parts = fen_string.split(" ")
        fen_board = parts[0].split('/')
        
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

        return image

    def normalize_evaluation(self, evaluation: str):
        '''returns centipawn evaluation normalized between 0 and 1'''
        
        # black has checkmate
        if '#' in evaluation and '-' in evaluation:
            evaluation = 0

        # white has checkmate
        elif '#' in evaluation and '+' in evaluation:
            evaluation = 1

        # if no checkmate, pass centipawn evaluation through sigmoid function
        else:
            try:
                evaluation = sigmoid(float(evaluation)/100)
            except:
                # if there is bad label, just set it to 0.5
                # this happens rarely (I think) so I'll just ignore it for now :p
                evaluation = 0.5

        return evaluation

    def generate_data(self, batch_IDs):
        '''generates the x and y data for all IDs within a mini batch.'''
        
        # initialize empty outputs
        x = np.empty((self.batch_size, *self.x_dim))
        y = np.empty((self.batch_size, *self.y_dim))

        # retrieve, format, and store data for every ID in the mini batch
        for i, ID in enumerate(batch_IDs):

            # get fen and evaluation
            fen = self.dataframe._get_value(ID, 'FEN')
            evaluation = self.dataframe._get_value(ID, 'Evaluation')

            # convert
            image = self.fen_to_image(fen)
            label = self.normalize_evaluation(evaluation)

            # store
            x[i,] = image
            y[i,] = label

        return x, y

    def __len__(self):
        'returns the number of batches per epoch'
        return int(np.floor(len(self.IDs) / self.batch_size))

    def __getitem__(self,index):
        '''returns x and y data for a mini batch'''

        # get the indexes and then IDS for the next minibatch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        temporary_list_IDS = [self.IDs[j] for j in indexes]

        # generate x and y datasets for the selected IDs
        x, y = self.generate_data(batch_IDs = temporary_list_IDS)

        return x,y