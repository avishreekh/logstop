from src.logstop import LTLFormula
from typing import Set

def parse_formula_from_string(formula_str: str) -> 'LTLFormula':
    """Parse grouped queries with Not/Next/Always/Eventually and And/Or/Until.

    Operators are case-insensitive. Binary precedence is Or, And, Until;
    parentheses preserve explicit grouping. Proposition names may contain spaces.
    """
    import re

    text = formula_str.lower().strip()
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced query parentheses")
    if depth:
        raise ValueError("Unbalanced query parentheses")

    def parse(text):
        text = text.strip()
        if not text:
            raise ValueError("Missing query operand")
        # Remove only parentheses enclosing the entire expression.
        while text.startswith("("):
            depth = 0
            closing = None
            for index, char in enumerate(text):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    closing = index
                    break
            if closing != len(text) - 1:
                break
            text = text[1:-1].strip()
            if not text:
                raise ValueError("Missing query operand")
        for operator in ("or", "and", "until"):
            depth = 0
            for index, char in enumerate(text):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif depth == 0:
                    match = re.match(r"\b" + operator + r"\b", text[index:])
                    if match and (index == 0 or text[index - 1].isspace()):
                        return LTLFormula(operator, parse(text[:index]),
                                          parse(text[index + len(operator):]))
        unary = re.match(r"^(not|next|always|eventually)\b", text)
        if unary:
            return LTLFormula(unary.group(1), parse(text[unary.end():]))
        if "(" in text or ")" in text:
            raise ValueError(f"Invalid query expression: {text!r}")
        if text in ("true", "false"):
            return LTLFormula(text.capitalize())
        return LTLFormula("class", text)

    return parse(text)


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