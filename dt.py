from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np
from numpy.typing import ArrayLike
from scorer import Scorer

class DecisionTree:
    """
    A decision tree classifier.

    Attributes:
        scorer (Scorer): The scorer used to evaluate the quality of a split.
        max_depth (int): The maximum depth of the tree.
        root (Node): The root node of the tree.
    """
    def __init__(self, scorer: Scorer, max_depth: int = 5) -> None:
        """
        Constructs a decision tree classifier.

        Parameters:
            scorer (Scorer): The scorer used to evaluate the quality of a split.
            max_depth (int): The maximum depth of the tree.

        Returns:
            None
        """
        self.scorer = scorer
        self.max_depth = max_depth
        self.root = None
    def fit(self, X, y):
        ...
    
    # Run
    if __name__ == '__main__':
        ...