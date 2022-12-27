import chess
from keras import models
import numpy as np
import time

class ChessEngine:

    def __init__(self, model_file_name:str,chess_board: chess.Board, engine_color: bool):

        self.board = chess_board
        self.model = models.load_model(model_file_name)
        self.engine_color = engine_color

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

    def get_board_evaluation(self) -> float:
        '''returns a number that corresponds to normalized board evaluation'''
        fen = self.board.fen()
        image = self.fen_to_image(fen)
        evaluation = self.model.predict(image,verbose = 0)
        return evaluation[0][0]


    def get_move(self) -> chess.Move:
        '''returns the best move'''

        if self.board.turn != self.engine_color:
            print("engine called at wrong time")
            raise ValueError

        evaluations = []
        eval_to_move = {}

        for move in self.board.legal_moves:

            self.board.push(move)

            evaluation = self.get_board_evaluation()
            evaluations.append(evaluation)
            eval_to_move[evaluation] = move

            self.board.pop()

        evaluations.sort()

        best_eval = evaluations[0] if self.engine_color == chess.BLACK else evaluations[-1]

        return eval_to_move[best_eval]

class Deep_ChessEngine2(ChessEngine):

    def __init__(self, model_file_name: str, chess_board: chess.Board, engine_color: bool, engine_depth = 2):
        super().__init__(model_file_name, chess_board, engine_color)
        self.depth = engine_depth

    def get_move(self) -> chess.Move:
        '''returns the best move with look ahead'''
        if self.board.turn != self.engine_color:
            print("engine called at wrong time")
            raise ValueError

        if self.depth == 0:
            time.sleep(5)

        ### dfs serach
        # init a dict that holds all the moves and look ahead evals we'll calculate
        best_move = None
        best_eval = None
        for move in self.board.legal_moves:
            move_eval = self.search_alpha_beta(depth = self.depth, move = move)

            # seems pretty redundant, but Idk a better way to write it more concisely AND understandable.
            if best_move == None:
                best_move = move
                best_eval = move_eval

            if self.engine_color == chess.WHITE and move_eval >= best_eval:
                best_move = move
                best_eval = move_eval

            elif self.engine_color == chess.BLACK and move_eval <= best_eval:
                best_move = move
                best_eval = move_eval

        return best_move
   

    def search(self, depth: int, move: chess.Move) -> float:
        '''returns the best evaluation of a move, given a certain serach depth.'''  
        global search_count
        search_count += 1

        print(move, self.board.turn)

        # if we reach the serach depth, evaluate the position and then unmake themove
        if depth == 0:
            self.board.push(move)
            evaluation = self.get_board_evaluation()
            self.board.pop()
            return evaluation

        elif self.board.turn == chess.BLACK:

            # play black's move
            self.board.push(move)

            # find white's best response, maxmizing
            maximum = -1000
            for sub_move in self.board.legal_moves:
                score  = self.search(depth-1,sub_move)
                if score > maximum:
                    maximum = score

            # undo black's move
            self.board.pop()

            return maximum

        elif self.board.turn == chess.WHITE:

            minimum = 1000
            self.board.push(move)
            for move in self.board.legal_moves:
        
                score  = self.search(depth-1,move)
                if score < minimum:
                    minimum = score
            self.board.pop()
            return minimum

    def search_alpha_beta(self, depth, move, alpha = -1000, beta = 1000):
        '''Same result as self.serach(), but with alpah beta optimization'''
        global ab_count
        ab_count += 1

        if depth == 0:
            self.board.push(move)
            evaluation = self.get_board_evaluation()
            self.board.pop()
            return evaluation

        elif self.board.turn == chess.BLACK:

            # play black's move
            self.board.push(move)
            # find white's best response, maxmizing
            score = -1000
            for sub_move in self.board.legal_moves:
                score  = max(score,self.search_alpha_beta(depth-1,sub_move, alpha, beta))
                if score > beta:
                    break

                alpha = max(alpha,score)

            # undo black's move
            self.board.pop()

            return score

        elif self.board.turn == chess.WHITE:

            self.board.push(move)

            score = 1000
            for move in self.board.legal_moves:
        
                score  = min(score,self.search_alpha_beta(depth-1,move,alpha, beta))
                if score < alpha:
                    break
                beta = min(beta,score)
            self.board.pop()
            return score

def test():
    chess_board = chess.Board()
    engine_color = chess.WHITE
    model_path = "/Users/williamzhang/Documents/College/chess project testing/OTB/Model_Cloud5_12_20_22"
    chess_engine = Deep_ChessEngine2(model_path, chess_board, engine_color, engine_depth=1)

    tic = time.perf_counter()
    print(chess_engine.get_move())
    toc = time.perf_counter()

    print(toc-tic)


