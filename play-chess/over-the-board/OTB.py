# Quick implementation of Chess. 
# Uses UCI move strings to make moves on the board. 
# Not very pretty.

import os
import sys
import chess

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import EngineClasses


def play_human_move(board: chess.Board):
    '''loop that allows the human player to select a move to play'''

    moves = [move.uci() for move in board.legal_moves]
    move = None

    print(f"Availble Moves: {moves.sort()}")
    
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


def play_engine_move(engine, board: chess.Board):
    '''gets and plays best move according to the engine used.'''

    move, move_eval, num_calls = engine.get_move()
    
    print(f"Engine plays {move.uci()}\nEval: {move_eval}\nPositions searched: {num_calls}")
    
    board.push(move)


def play_chess():
    '''play_chess speaks for itself - hans nieman'''

    print('Welcome to Chess')

    # pick sides
    human_color = None

    while human_color == None:
        side_selection = input("Pick a side to play (white or black) : ")

        if side_selection.lower() in ['black','b']:
            human_color = chess.BLACK

        elif side_selection.lower() in ['white','w']:
            human_color = chess.WHITE

    engine_color = not human_color


    # initialize board and engine
    chess_board = chess.Board()
    engine_depth = int(input("Engine depth (0-4): "))
    use_ai = True if input("Use AI evaluation? (y/n): ").lower() in ['yes, y, ye'] else False

    params = {
        'board' : chess_board,
        'depth' : engine_depth,
        'model_file_name' : "../keras-exports/Model_Cloud5_12_20_22",
    }

    if use_ai:
        chess_engine - EngineClasses.AiEngine(**params)
    else:
        chess_engine = EngineClasses.SimpleChessEngine(chess_board, engine_depth)

    # game loop
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