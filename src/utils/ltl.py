from src.logstop import LTLFormula
from typing import Set

def parse_formula_from_string(formula_str: str) -> 'LTLFormula':
    """
    Parse a simple LTL formula from a string.
    Args:
        formula_str (str): The LTL formula as a string.
    Returns:
        LTLFormula: The parsed LTL formula object.

    Example usage:
        phi = LTLFormula.parse_formula_from_string("(p until q)")
        phi = LTLFormula.parse_formula_from_string("always (p)")
        phi = LTLFormula.parse_formula_from_string("(p and (not q))")
    """
    # get the main operator and split accordingly
    formula_str = formula_str.lower().strip()
    # the main operator is the first one not in parentheses
    def find_main_operator(s: str):
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0:
                # check for operators
                for op in [" until ", " and ", " or ", "next ", "always ", "eventually "]:
                    if s.startswith(op, i):
                        return op.strip(), i
                if s.startswith("not ", i):
                    return "not", i
        return None, -1
    op, idx = find_main_operator(formula_str)
    # print(f"Parsing formula: {formula_str}, found op: {op} at idx: {idx}")
    if op is None:
        # base case: either True, False, or class
        if formula_str == "True":
            return LTLFormula("True")
        elif formula_str == "False":
            return LTLFormula("False")
        else:
            return LTLFormula("class", formula_str)
    if op in ["until", "and", "or"]:
        left_str = formula_str[:idx].strip().lstrip('(').rstrip(')')
        right_str = formula_str[idx + len(op) + 1:].strip().lstrip('(').rstrip(')')
        left_formula = parse_formula_from_string(left_str)
        right_formula = parse_formula_from_string(right_str)
        return LTLFormula(op, left_formula, right_formula)
    elif op in ["next", "always", "eventually", "not"]:
        subformula_str = formula_str[idx + len(op):].strip().lstrip('(').rstrip(')')
        subformula = parse_formula_from_string(subformula_str)
        return LTLFormula(op, subformula)
    else:
        raise ValueError(f"Unknown operator in formula: {op}")
        
def extract_local_properties(ltl_formula: LTLFormula) -> Set[str]:
    """
    Recursively extract all local properties (class names) from the LTL formula.
    Returns:
        Set[str]: A set of local property names used in the formula.
    """
    local_props = set()
    if ltl_formula.op == "class":
        local_props.add(ltl_formula.left)
    elif ltl_formula.op in ["not", "next", "always", "eventually"]:
        local_props.update(extract_local_properties(ltl_formula.left))
    elif ltl_formula.op in ["and", "or", "until"]:
        local_props.update(extract_local_properties(ltl_formula.left))
        local_props.update(extract_local_properties(ltl_formula.right))
    return local_props