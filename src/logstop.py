import math
from typing import Dict


class LTLFormula:
    def __init__(self, op, left=None, right=None):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        if self.op == "True":
            return "True"
        elif self.op == "False":
            return "False"
        elif self.op == "class":
            return str(self.left)   # local formula
        elif self.op == "not":
            return f"not {self.left}"
        elif self.op == "and":
            return f"{self.left} and {self.right}"
        elif self.op == "or":
            return f"{self.left} or {self.right}"
        elif self.op == "next":
            return f"next {self.left}"
        elif self.op == "until":
            return f"{self.left} until {self.right}"
        elif self.op == "always":
            return f"always {self.left}"
        elif self.op == "eventually":
            return f"eventually {self.left}"
        else:
            raise ValueError(f"Unknown operator: {self.op}")
        
def log(x):
    if x <= 0.0:
        return float("-inf")
    return math.log(x)

def exp(x):
    if x == float("-inf"):
        return 0.0
    return math.exp(x)
        
def logstop(trace : Dict[str, list], phi : LTLFormula, start_idx: int, end_idx: int, w: int = 1, memo={}) -> float:
    """
    Compute LogSTOP for the given trace and formula phi over the interval [start_idx, end_idx].
    Args:
        trace (Dict[str, list]): The input trace as a dictionary mapping variable names to [0,1] predictions over time.
        phi (LTLFormula): The LTL formula to evaluate.
        start_idx (int): The starting index of the interval.
        end_idx (int): The ending index of the interval.
        w (int): The downsampling smoothing window (default is 1, meaning no smoothing).
        memo (dict): A memoization dictionary to cache results (default is None).
    Returns:
        float: The LogSTOP value for the formula over the specified interval.

    Example usage:
        trace = {"p": [0.1, 0.5, 0.9, 0.7], "q": [0.2, 0.8, 0.4, 0.6]}
        phi = LTLFormula("eventually", LTLFormula("class", "p"))    # Eventually p
        result = logstop(trace, phi, 0, 3)                      # Evaluate from index 0 to 3    
    """
    if memo is not None and (phi, start_idx) in memo:
        return memo[(phi, start_idx)]
    
    if start_idx > end_idx:
        return log(0.0)
    if phi.op == "True":
        return log(1.0)
    elif phi.op == "False":
        return log(0.0)
    elif phi.op == "class":
        # Local formula evaluation (averaged over start_idx to start_idx + w - 1)
        window_length = min(w, end_idx - start_idx + 1)
        prob = sum(trace[phi.left][start_idx:start_idx + window_length]) / window_length
        result = log(prob)
    elif phi.op == "not":
        result = log(1.0 - exp(logstop(trace, phi.left, start_idx, end_idx, w, memo)))
    elif phi.op == "and":
        left_result = logstop(trace, phi.left, start_idx, end_idx, w, memo)
        right_result = logstop(trace, phi.right, start_idx, end_idx, w, memo)
        result = left_result + right_result
    elif phi.op == "or":
        # De Morgan's law: A or B = not (not A and not B)
        phi_or = LTLFormula("not", 
                            LTLFormula("and", 
                                       LTLFormula("not", phi.left), 
                                       LTLFormula("not", phi.right)))
        result = logstop(trace, phi_or, start_idx, end_idx, w, memo)
    elif phi.op == "next":
        if start_idx + w > end_idx:
            return log(0.0) # next is out of bounds
        result = logstop(trace, phi.left, start_idx + w, end_idx, w, memo)
    elif phi.op == "until":
        # phi1 until phi2 = phi2 or (phi1 and not phi2 and next (phi1 until phi2))
        # The or branches are mutually exclusive
        phi_next = LTLFormula("and",
                                phi.left,
                                LTLFormula("and",
                                            LTLFormula("not", phi.right),
                                            LTLFormula("next", phi))
                            )
        result_phi2 = logstop(trace, phi.right, start_idx, end_idx, w, memo)
        result_phi_next = logstop(trace, phi_next, start_idx, end_idx, w, memo)
        prob = exp(result_phi2) + exp(result_phi_next)     # mutual exclusivity
        result = log(prob)
    elif phi.op == "always":
        # always phi = phi and next always phi
        # could also use: always phi = not (eventually (not phi))
        if start_idx + w > end_idx: # last block (next is out of bounds)
            result = logstop(trace, phi.left, start_idx, end_idx, w, memo)  
        else:
            phi_always = LTLFormula("and", phi.left, LTLFormula("next", phi))
            result = logstop(trace, phi_always, start_idx, end_idx, w, memo)
    elif phi.op == "eventually":
        # eventually phi = True until phi
        # could also use: eventually phi = not (always (not phi))
        phi_eventually = LTLFormula("until", LTLFormula("True"), phi.left)
        result = logstop(trace, phi_eventually, start_idx, end_idx, w, memo)
    else:
        raise ValueError(f"Unknown operator: {phi.op}")
    
    if memo is not None:
        memo[(phi, start_idx)] = result
    return result

    