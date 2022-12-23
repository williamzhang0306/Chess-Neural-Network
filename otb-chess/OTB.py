import chess
from EngineClass import *

welcome_message = "Welcome to cheses"

def play_human_move(board):
    print("Availble Moves:")
    moves = [move.uci() for move in board.legal_moves]
    moves.sort()
    print(moves)

    move = None
    while move == None:
        selection_str = input("\nEnter UCI move: ").strip()
        try:
            move = chess.Move.from_uci(selection_str)
        except:
            print('Move not valid UCI string, try again')

        if move not in board.legal_moves:
            move = None
            print('Move is not legal, try again')
    print("\nHuman plays", move.uci())
    board.push(move)

def play_engine_move(engine, board):
    move = engine.get_move()
    print("Engine plays", move.uci())
    board.push(move)

def play_chess():

    ## welcome
    print(welcome_message)


    ## let human pick sides
    human_color = None

    while human_color == None:
        side_selection = input("Pick a side to play (white or black) : ")

        if side_selection.lower() in ['black','b']:
            human_color = chess.BLACK

        elif side_selection.lower() in ['white','w']:
            human_color = chess.WHITE

    engine_color = not human_color

    engine_depth = int(input("Engine Depth (0-3): "))

    # initialize board and engine
    chess_board = chess.Board()
    params = {
        'chess_board' : chess_board,
        'engine_color' : engine_color,
        'model_file_name' : "/Users/williamzhang/Documents/College/Neural-Network-Chess/keras-exports/Model_Cloud5_12_20_22",
        'engine_depth' : engine_depth
    }

    chess_engine = Deep_ChessEngine2(**params)

    # gaeme loop
    current_color = chess.WHITE
    print("\n",chess_board,"\n")

    while chess_board.outcome() == None:

        if human_color == current_color:
            play_human_move(chess_board)

        elif engine_color == current_color:
            play_engine_move(chess_engine, chess_board)

        print("\n",chess_board,"\n")

        # flip from white to black vice versa
        current_color = not current_color

    print('Outcome: ', chess_board.outcome())

if __name__ == '__main__':
    play_chess()