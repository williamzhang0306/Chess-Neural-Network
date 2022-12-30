import chess
import numpy as np
import keras
import time
import sys


class SimpleChessEngine():
  # classic chess engine with alpha beta and move ordering optimization

  def __init__(self, board: chess.Board, depth = 0):
    self.board = board
    self.depth = depth
    self.count = 0

  def evaluate(self) -> int:
    '''Evaluates the board's current states. This implementation just counts material.'''

    # check for checkmates and draws
    outcome = self.board.outcome()
    if outcome != None:
      # stalemate
      if outcome.winner == None: return 0

      # checkmates
      if outcome.winner == self.board.turn: return 6969

      if outcome.winner != self.board.turn: return -6969

    evaluation = 0

    piece_value = {
      chess.PAWN : 1,
      chess.KNIGHT : 3,
      chess.BISHOP : 3,
      chess.ROOK : 5,
      chess.QUEEN : 9,
      chess.KING : 0
    }

    for square in self.board.piece_map():
      piece = self.board.piece_at(square)

      if piece == None: continue

      points = piece_value[piece.piece_type]

      if piece.color == chess.BLACK:
        points *= -1

      evaluation += points

    perspective = 1 if self.board.turn == chess.WHITE else -1

    return evaluation * perspective

  def ordered_moves(self) -> list:
    '''Returns a reordered list of legal moves. 
    Moves with promotions or captures are at front of the list.'''

    ordered_list = []

    for move in self.board.legal_moves:

        if move.drop != None or move.promotion != None:
          ordered_list.insert(0,move)

        else:
          ordered_list.append(move)
    #print([move.uci() for move in ordered_list])
    return ordered_list

  def nega_max_alpha_beta(self, depth, alpha = -1 *sys.maxsize, beta = sys.maxsize):
    '''nega_max serach with alpha beta optimization'''
    
    self.count += 1

    # evaluate if depth reached or no legal moves
    if depth == 0 or self.ordered_moves() == []: return self.evaluate()

    # iterate through all moves
    for move in self.ordered_moves():
      # recursively call to get relative evaluation
      self.board.push(move)
      score = -1 * self.nega_max_alpha_beta(depth=depth-1, alpha = -1 * beta, beta = -1 * alpha)
      self.board.pop()

      if score >= beta:
        return beta

      if score > alpha:
        alpha = score

    return alpha

  def get_move(self) -> chess.Move:
    '''returns the best move according to the engine'''
    self.count = 0
    moves = []
    values = []

    for move in self.ordered_moves():
      self.board.push(move)
      value = -1 * self.nega_max_alpha_beta(depth = self.depth)
      self.board.pop()

      moves.append(move)
      values.append(value)

    return moves[values.index(max(values))],max(values), self.count


class AiEngine(SimpleChessEngine):

  def __init__(self, model: str, board: chess.Board, depth = 0):
    super().__init__(board, depth)
    self.keras_model = keras.models.load_model(model)

  def fen_to_image(self, fen_string: str) -> np.array:
    '''Converts fen string to 8x8x12 (rows x cols x channels) image. 
    13 channels represent 12 pieces plue one channel for white or black is playing.
    In each pixel of the image, 1 represents a white piece is present, 
    -1 black piece present, 0 no piece present.'''
    # used for conversion
    piece_to_channel = {
      'P':0, 'N':1, 'B':2, 'R':3 , 'Q':4 , 'K':5,
      'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11
    }
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

  def evaluate(self) -> float:
      '''returns a number that corresponds to normalized board evaluation'''
      fen = self.board.fen()
      image = self.fen_to_image(fen)
      evaluation = self.keras_model.predict(image,verbose = 0)[0][0] - 0.5
      perspective = 1 if self.board.turn == chess.WHITE else -1

      return evaluation * perspective