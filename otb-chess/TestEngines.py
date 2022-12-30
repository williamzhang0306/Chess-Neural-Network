import time
import chess
from EngineClasses import *

count = 0

def test():

  global count

  test_board = chess.Board()

  engine = SimpleChessEngine(test_board, 4)

  print("\nno opitmizaiton")
  count = 0
  time.sleep(3)
  tic = time.perf_counter()
  print(engine.get_move(use_ab = True))
  toc = time.perf_counter()
  print(toc-tic, count)

  print("\njust ab pruning")
  count = 0
  time.sleep(3)
  tic = time.perf_counter()
  print(engine.get_move(use_ab = False))
  toc = time.perf_counter()
  print(toc-tic, count)

  print("\nalpba beta pruning and move order optimization")
  count = 0
  time.sleep(3)
  tic = time.perf_counter()
  print(engine.get_move(use_ab = True))
  toc = time.perf_counter()
  print(toc-tic, count)

def test_ai():

  # initialize test conditions and engine
  global count
  test_board = chess.Board("4R3/2P2P2/4B3/3r1pN1/1Q3P1B/5P1P/8/k1KR3N b - - 0 1")
  model_path = "/Users/williamzhang/Documents/College/Neural-Network-Chess/keras-exports/Model_Cloud5_12_20_22"

  params = {
    'model' : model_path,
    'board' : test_board,
    'depth' : 2
  }

  engine = AiEngine(**params)

  #no optimizaiton test
  print("\n AI eval with no opitmizaiton")
  count = 0
  time.sleep(3)
  tic = time.perf_counter()
  print(engine.get_move(use_ab = False))
  toc = time.perf_counter()
  print(toc-tic, count)  

  # optimization test
  print("\nAI eval with alpba beta pruning and move order optimization")
  count = 0
  time.sleep(3)
  tic = time.perf_counter()
  print(engine.get_move(use_ab = True))
  toc = time.perf_counter()
  print(toc-tic, count)


if __name__ == '__main__':
  test()
  test_ai()