import collections
from typing import Any, Dict, Sequence, Set, Tuple

import numpy as np
from numpy.typing import ArrayLike

class Scorer:
    """
    This class represents a scorer for a decision tree.

    Attributes:
        class_labels (ArrayLike): A list of the class labels.
        alpha (int): The alpha value for Laplace smoothing.

    """
    def __init__(self, type: str, class_labels: Sequence, alpha: int = 1) -> None:
        """
        The constructor for the Scorer class. Saves the class labels to
        `self.class_labels` and the alpha value to `self.alpha`.

        Parameters:
            type (str): The type of scorer to use. Either "information" or "gini".
            class_labels (Sequence): A list or set of unique class labels.
            alpha (int): The alpha value for Laplace smoothing.

        Returns:
            None

        Examples:
            >>> scorer = Scorer("information", ["A", "B"])
            >>> scorer.type
            'information'
            >>> sorted(scorer.class_labels)
            ['A', 'B']
            >>> scorer.alpha
            1
        """

        if type not in ["information", "gini"]:
            raise ValueError("type must be either 'information' or 'gini'")
        
        # >>> YOUR CODE HERE >>>
        self.type = type
        self.class_labels = class_labels
        self.alpha = alpha
        # <<< END OF YOUR CODE <<<
    
    
    #This function calculates the score for a set of labels.
    def  score(self, labels: ArrayLike) -> float:

        if self.type == "information":
            return self.information_score(labels)
        elif self.type == "gini":
            return self.gini_score(labels)
        
        raise ValueError("type must be either 'information' or 'gini'")
    
    #This function calculates the gain for a set of labels and a split attribute.
    def gain(self, data: ArrayLike, labels: ArrayLike, split_attribute: int) -> float:
        

        if self.type == "information":
            return self.information_gain(data, labels, split_attribute)
        elif self.type == "gini":
            return self.gini_gain(data, labels, split_attribute)
        
        raise ValueError("type must be either 'information' or 'gini'")