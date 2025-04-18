from collections import Counter
from typing import Dict, List
from typing import Dict

import numpy as np
from javalang.tokenizer import JavaToken
from javalang.tree import MethodDeclaration
from javalang.tree import Node
from javalang.tree import TernaryExpression

from .utils import get_nodes
from .utils import get_nodes_count
from .utils import identifiers
from .utils import keywords
from .utils import literals
from .utils import non_empty_lines

# Calculates term frequency (TF) of each identifier (e.g., variable names) in the code
def calculate_WordUnigramTF(tokens: List[JavaToken]) -> Dict:
    values = map(lambda it: it.value, identifiers(tokens))
    count = Counter(values)

    features = {}
    total_count = sum(count.values())
    for key, value in count.items():
        features[f'WordUnigramTF_{key}'] = value / total_count

    return features

# Calculates log-scaled frequency of each Java keyword relative to file length
def calculate_NumKeyword(tokens: List[JavaToken], file_length: int) -> Dict:
    values = map(lambda it: it.value, keywords(tokens))
    count = Counter(values)

    features = {}
    for key, value in count.items():
        features[f'ln(num_{key}/length)'] = np.log(value / file_length)

    return features

# Calculates log-scaled frequency of each Java keyword relative to file length
def calculate_NumTokens(tokens: List[JavaToken], file_length: int) -> Dict:
    num_identifiers = len(identifiers(tokens))
    value = np.log(num_identifiers / file_length)
    return {'ln(numTokens/length)': value}

# Counts the number of comment lines and normalizes by total file length (in characters)
def calculate_NumComments(code: str) -> Dict:
    lines = non_empty_lines(code)
    num_comments = sum(line.strip()[:2] == '//' for line in lines)
    value = np.log(num_comments / len(code))
    return {'ln(numComments/length)': value}

# Computes number of literals (e.g., numbers, strings) normalized by file length
def calculate_NumLiterals(tokens: List[JavaToken], file_length: int) -> Dict:
    num_literals = len(literals(tokens))
    value = np.log(num_literals / file_length)
    return {'ln(numLiterals/length)': value}

# Redundant with calculate_NumKeyword (same purpose) — consider merging or renaming for clarity
def calculate_NumKeywords(tokens: List[JavaToken], file_length: int) -> Dict:
    num_literals = len(keywords(tokens))
    value = np.log(num_literals / file_length)
    return {'ln(numKeywords/length)': value}

# Counts number of function (method) declarations and normalizes by file length
def calculate_NumFunctions(tree: Node, file_length: int) -> Dict:
    num_functions = get_nodes_count(tree, MethodDeclaration)
    value = np.log(num_functions / file_length)
    return {'ln(numFunctions/length)': value}

# Detects ternary (condition ? true : false) usage frequency
def calculate_NumTernary(tree: Node, file_length: int) -> Dict:
    num_ternary = get_nodes_count(tree, TernaryExpression)
    value = np.log(num_ternary / file_length)
    return {'ln(numTernary/length)': value}

# Calculates the average length of lines in the source code
def calculate_AvgLineLength(code: str) -> Dict:
    lines = code.split('\n')
    value = np.mean([len(line) for line in lines])
    return {'avgLineLength': value}

# Calculates standard deviation of line lengths to measure code layout variability
def calculate_StdDevLineLength(code: str) -> Dict:
    lines = code.split('\n')
    value = np.std([len(line) for line in lines])
    return {'stdDevLineLength': value}

# Calculates the average number of parameters per method
def calculate_AvgParams(tree: Node) -> Dict:
    nodes = get_nodes(tree, MethodDeclaration)
    num_params = [len(node.children[6]) for node in nodes]
    value = np.mean(num_params)
    return {'avgParams': value}

# Computes standard deviation in number of parameters per method
def calculate_StdDevNumParams(tree: Node) -> Dict:
    nodes = get_nodes(tree, MethodDeclaration)
    num_params = [len(node.children[6]) for node in nodes]
    value = np.std(num_params)
    return {'stdDevNumParams': value}