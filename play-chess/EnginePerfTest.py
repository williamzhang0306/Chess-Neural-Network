import chess
import time
from EngineClasses import *

class TestEngines():

  def __init__(self):
    print('init')
    board = chess.Board()
    
    model_path = '/Users/williamzhang/Documents/College/Neural-Network-Chess/keras-exports/Model_Cloud9_12_29_22'
    params = {
      'depth': 4,
      'board': board,
      'model': model_path
    }
    self.TfLiteEngine = TFLiteEngine(**params)
    self.AiEngine = AiEngine(**params)
    del params['model'] # simple engine doesn't take model paramater
    self.SimpleEvalEngine = SimpleChessEngine(**params)

    print('\nInitializaiton Done\n')

  def timer(func):
    'simple function timer'
    def timed_func(*args, **kwargs):
      time.sleep(3)
      tic = time.perf_counter()
      func(*args, **kwargs)
      toc = time.perf_counter()
      print('Time Elapsed:',toc-tic)

    return timed_func

  def describe_test(test_func):
    def described_test(self,engine):
      print('\n' + engine.evaluation_type +" "+ test_func.__doc__)
      test_func(self,engine)

    return described_test

  @timer
  @describe_test
  def evaluate_one_position(self,engine: SimpleChessEngine):
    'evaluating just one position'
    evaluation = engine.evaluate()
    print('Evaluation:', evaluation)

  @timer
  @describe_test
  def evaluation_to_depth(self,engine: SimpleChessEngine):
    'evaluation used to depth'
    move, evaluation, searches = engine.get_move()
    print('Move:',move)
    print('Evaluation:', evaluation)
    print('Searches:', searches)

  def run_test(self):
    engines = [self.SimpleEvalEngine, self.TfLiteEngine, self.AiEngine]

    # for tested_engine in engines:
    #   self.evaluate_one_position(engine = tested_engine)

    for tested_engine in engines:
      self.evaluation_to_depth(engine = tested_engine)

def main(): 
  e = TestEngines()
  e.run_test()

main()