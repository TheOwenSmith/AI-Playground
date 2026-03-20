import random
import matplotlib.pyplot as plt
import numpy as np

class Game:
  def __init__(self):
    self.board = [[0] * 4 for _ in range(4)]
    self.score = 0
  
  def has_empty(self):
    for i in range(4):
      for j in range(4):
        if self.board[i][j] == 0:
          return True
    return False
  
  def add_random(self):
    if not self.has_empty():
      return
    
    while True:
      i = random.randint(0, 3)
      j = random.randint(0, 3)
      if self.board[i][j] == 0:
        self.board[i][j] = random.choice([1, 2])
        return
  
  def cell_to_string(_, cell):
    return '0' if cell == 0 else str(2 ** cell)
  
  def print_board(self):
    stringified_board = '-' * 5 + str(self.score) + '-' * 5 + '\n' + '\n'.join([' '.join(self.cell_to_string(cell) for cell in row) for row in self.board]) + '\n' + '-' * 10
    print(stringified_board)
  
  @staticmethod
  def prompt_move(available_moves):
    user_input = input(f'Which way do you want to move ({', '.join(available_moves)}): ')
    while user_input not in available_moves:
      print(f'Invalid input \'{user_input}\'!')
      user_input = input(f'Which way do you want to move ({', '.join(available_moves)}): ')
    return user_input
  
  def get_available_moves(self):
    available_moves = []
    for choice in ['left', 'right', 'up', 'down']:
      has_changed = self.make_move(choice, modify_board=False)
      if has_changed:
        available_moves.append(choice)
    return available_moves
  
  def make_move(self, move, modify_board=True):
    has_changed = False
    if move == 'left':
      #left
      for i in range(4):
        new_row = []
        last = -1
        for j in range(4):
          cell = self.board[i][j]
          if cell == 0:
            continue
          elif cell == last:
            if modify_board:
              self.score += 2 ** (cell + 1)
            new_row.append(cell + 1)
            last = -1
          elif last != -1:
            new_row.append(last)
            last = cell
          else:
            last = cell
          
        if last != -1:
          new_row.append(last)
        new_row += [0] * (4 - len(new_row))

        if new_row != self.board[i]:
          if modify_board:
            self.board[i] = new_row
          has_changed = True
    elif move == 'right':
      # right
      for i in range(4):
        new_row = []
        last = -1
        for j in range(3, -1, -1):
          cell = self.board[i][j]
          if cell == 0:
            continue
          elif cell == last:
            if modify_board:
              self.score += 2 ** (cell + 1)
            new_row.insert(0, cell + 1)
            last = -1
          elif last != -1:
            new_row.insert(0, last)
            last = cell
          else:
            last = cell
          
        if last != -1:
          new_row.insert(0, last)
        new_row = [0] * (4 - len(new_row)) + new_row

        if new_row != self.board[i]:
          if modify_board:
            self.board[i] = new_row
          has_changed = True
    elif move == 'up':
      # up
      for j in range(4):
        new_column = []
        last = -1
        for i in range(4):
          cell = self.board[i][j]
          if cell == 0:
            continue
          elif cell == last:
            if modify_board:
              self.score += 2 ** (cell + 1)
            new_column.append(cell + 1)
            last = -1
          elif last != -1:
            new_column.append(last)
            last = cell
          else:
            last = cell
          
        if last != -1:
          new_column.append(last)
        new_column += [0] * (4 - len(new_column))

        if new_column != [self.board[i][j] for i in range(4)]:
          if modify_board:
            for i in range(4):
              self.board[i][j] = new_column[i]
          has_changed = True
    elif move == 'down':
      # down
      for j in range(4):
        new_column = []
        last = -1
        for i in range(3, -1, -1):
          cell = self.board[i][j]
          if cell == 0:
            continue
          elif cell == last:
            if modify_board:
              self.score += 2 ** (cell + 1)
            new_column.insert(0, cell + 1)
            last = -1
          elif last != -1:
            new_column.insert(0, last)
            last = cell
          else:
            last = cell
          
        if last != -1:
          new_column.insert(0, last)
        new_column = [0] * (4 - len(new_column)) + new_column

        if new_column != [self.board[i][j] for i in range(4)]:
          if modify_board:
            for i in range(4):
              self.board[i][j] = new_column[i]
          has_changed = True
    return has_changed

  def play(self, prompt_fn=prompt_move, log=True):
    while True:
      self.add_random()
      if log:
        self.print_board()
      
      available_moves = self.get_available_moves()
      if len(available_moves) == 0:
        return self.score
      
      user_input = prompt_fn(available_moves)
      self.make_move(user_input)

def agent_fn(available_moves):
  return random.choice(available_moves)

def main():
  n = 100
  final_scores = []
  for i in range(1, 100 + 1):
    if i % 100 == 0:
      print(f'Game: {i}')
    
    final_score_sum = 0
    for _ in range(n):
      g = Game()
      final_score_sum += g.play(prompt_fn=agent_fn, log=False)
    final_scores.append(final_score_sum / n)
  
  # Cursor generated
  plt.figure(figsize=(12, 6))
  plt.hist(final_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
  plt.xlabel('Final Score', fontsize=12, fontweight='bold')
  plt.ylabel('Frequency', fontsize=12, fontweight='bold')
  plt.title(f'Distribution of Final Scores Across 100 Games (Sample Size {n})', fontsize=14, fontweight='bold', pad=20)
  plt.grid(True, alpha=0.3, linestyle='--')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
