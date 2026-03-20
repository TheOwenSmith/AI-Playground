import random
import numpy as np
from typing import List, Tuple, Optional, Callable

class Game:
  def __init__(self):
    self.board = np.zeros((4, 4), dtype=np.int32)
    self.score = 0
    self._empty_cells = [(i, j) for i in range(4) for j in range(4)]
  
  def has_empty(self) -> bool:
    """Check if there are empty cells - optimized using empty_cells cache."""
    return len(self._empty_cells) > 0
  
  def add_random(self):
    """Add a random tile (2 or 4) to an empty cell - optimized."""
    if not self._empty_cells:
      return
    
    # Pick random empty cell and remove from cache
    idx = random.randrange(len(self._empty_cells))
    i, j = self._empty_cells.pop(idx)
    
    # 90% chance of 2, 10% chance of 4 (standard 2048 rules)
    self.board[i, j] = random.choices([1, 2], weights=[9, 1])[0]
  
  def _slide_row(self, row: np.ndarray, modify_board: bool = True) -> Tuple[np.ndarray, bool, int]:
    """Slide and merge a single row. Returns (new_row, changed, score_delta)."""
    # Remove zeros
    non_zeros = row[row != 0]
    if len(non_zeros) == 0:
      return np.zeros(4, dtype=np.int32), False, 0
    
    # Merge adjacent equal values
    new_row = []
    score_delta = 0
    i = 0
    while i < len(non_zeros):
      if i < len(non_zeros) - 1 and non_zeros[i] == non_zeros[i + 1]:
        # Merge
        merged = non_zeros[i] + 1
        new_row.append(merged)
        if modify_board:
          score_delta += 2 ** merged
        i += 2
      else:
        new_row.append(non_zeros[i])
        i += 1
    
    # Pad with zeros
    new_row = np.array(new_row + [0] * (4 - len(new_row)), dtype=np.int32)
    changed = not np.array_equal(row, new_row)
    
    return new_row, changed, score_delta
  
  def make_move(self, move: str, modify_board: bool = True) -> bool:
    """Make a move in the specified direction. Returns True if board changed."""
    has_changed = False
    total_score = 0
    
    if move == 'left':
      new_board = np.zeros_like(self.board)
      for i in range(4):
        new_row, changed, score_delta = self._slide_row(self.board[i], modify_board)
        new_board[i] = new_row
        if changed:
          has_changed = True
          total_score += score_delta
      
    elif move == 'right':
      new_board = np.zeros_like(self.board)
      for i in range(4):
        flipped_row = np.flip(self.board[i])
        new_row, changed, score_delta = self._slide_row(flipped_row, modify_board)
        new_board[i] = np.flip(new_row)
        if changed:
          has_changed = True
          total_score += score_delta
      
    elif move == 'up':
      new_board = np.zeros_like(self.board)
      for j in range(4):
        col = self.board[:, j]
        new_col, changed, score_delta = self._slide_row(col, modify_board)
        new_board[:, j] = new_col
        if changed:
          has_changed = True
          total_score += score_delta
      
    elif move == 'down':
      new_board = np.zeros_like(self.board)
      for j in range(4):
        flipped_col = np.flip(self.board[:, j])
        new_col, changed, score_delta = self._slide_row(flipped_col, modify_board)
        new_board[:, j] = np.flip(new_col)
        if changed:
          has_changed = True
          total_score += score_delta
    else:
      return False
    
    if has_changed and modify_board:
      self.board = new_board
      self._update_empty_cells()
      self.score += total_score
    
    return has_changed
  
  def _update_empty_cells(self):
    """Update the cache of empty cell positions."""
    self._empty_cells = [(i, j) for i in range(4) for j in range(4) 
                         if self.board[i, j] == 0]
  
  def get_available_moves(self) -> List[str]:
    """Get list of available moves."""
    available_moves = []
    for move in ['left', 'right', 'up', 'down']:
      if self.make_move(move, modify_board=False):
        available_moves.append(move)
    return available_moves
  
  def get_state(self) -> np.ndarray:
    """Get current board state as numpy array (for AI input)."""
    return self.board.copy()
  
  def get_features(self) -> np.ndarray:
    """Extract features for AI: board values, empty cells, monotonicity, smoothness."""
    features = []
    
    # Board values (flattened)
    features.extend(self.board.flatten())
    
    # Number of empty cells
    features.append(len(self._empty_cells))
    
    # Monotonicity (prefer increasing/decreasing rows/columns)
    monotonicity = 0
    for i in range(4):
      row = self.board[i]
      if all(row[j] <= row[j+1] for j in range(3) if row[j] != 0):
        monotonicity += 1
      if all(row[j] >= row[j+1] for j in range(3) if row[j] != 0):
        monotonicity += 1
    for j in range(4):
      col = self.board[:, j]
      if all(col[i] <= col[i+1] for i in range(3) if col[i] != 0):
        monotonicity += 1
      if all(col[i] >= col[i+1] for i in range(3) if col[i] != 0):
        monotonicity += 1
    features.append(monotonicity)
    
    # Smoothness (sum of differences between adjacent cells)
    smoothness = 0
    for i in range(4):
      for j in range(3):
        if self.board[i, j] != 0 and self.board[i, j+1] != 0:
          smoothness -= abs(self.board[i, j] - self.board[i, j+1])
    for i in range(3):
      for j in range(4):
        if self.board[i, j] != 0 and self.board[i+1, j] != 0:
          smoothness -= abs(self.board[i, j] - self.board[i+1, j])
    features.append(smoothness)
    
    # Max tile value
    features.append(self.board.max())
    
    return np.array(features, dtype=np.float32)
  
  def cell_to_string(self, cell: int) -> str:
    """Convert cell value to string representation."""
    return '0' if cell == 0 else str(2 ** cell)
  
  def print_board(self):
    """Print the current board state."""
    stringified_board = '-' * 5 + str(self.score) + '-' * 5 + '\n'
    stringified_board += '\n'.join([' '.join(self.cell_to_string(cell) for cell in row) 
                                   for row in self.board]) + '\n' + '-' * 10
    print(stringified_board)
  
  @staticmethod
  def prompt_move(available_moves: List[str]) -> str:
    """Prompt user for move input."""
    user_input = input(f'Which way do you want to move ({", ".join(available_moves)}): ')
    while user_input not in available_moves:
      print(f'Invalid input \'{user_input}\'!')
      user_input = input(f'Which way do you want to move ({", ".join(available_moves)}): ')
    return user_input
  
  def play(self, prompt_fn: Callable = None, log: bool = True) -> int:
    """Play the game until game over. Returns final score."""
    if prompt_fn is None:
      prompt_fn = self.prompt_move
    
    while True:
      self.add_random()
      if log:
        self.print_board()
      
      available_moves = self.get_available_moves()
      if len(available_moves) == 0:
        return self.score
      
      user_input = prompt_fn(available_moves)
      self.make_move(user_input)
  
  def copy(self) -> 'Game':
    """Create a deep copy of the game state."""
    new_game = Game()
    new_game.board = self.board.copy()
    new_game.score = self.score
    new_game._empty_cells = self._empty_cells.copy()
    return new_game

def main():
  pass

if __name__ == '__main__':
  main()
