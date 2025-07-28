import logging
import svg
from dataclasses import dataclass

@dataclass
class PositionerState:
    name: str
    position: int
    number_positions: int
    svg: int

class PositionerBase:
    """
    Positioner base representation. Discrete (integer) values only
    """

    def __init__(self,
                 n_positions: int,
                 position: int = 1,
                 name: str = '') -> None:
        self.name = name

        self.n_positions = n_positions
        self.position = None
        self.move(position)
    
    def move(self, position: int) -> None:
        """Move valve to position

        Args:
            position (int): desired position
        """
        if self.validate_move(position):
            self.position = position
    
    def validate_move(self, position: int) -> bool:
        """Validate a move to a desired position

        Args:
            position (int): desired position

        Returns:
            bool: True if valid, False if not
        """

        if position not in range(0, self.n_positions + 1):
            logging.error(f'{self.name} Move validation error: requested position {position} is not an integer between 0 (off) and {self.n_positions}')
            return False
        
        return True