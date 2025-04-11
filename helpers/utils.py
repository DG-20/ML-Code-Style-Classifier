from typing import Dict
from typing import Iterable
from typing import List

from javalang.tokenizer import JavaToken, Identifier, Keyword, Literal
from javalang.tree import Node

# Returns a list of Identifier tokens from the token list
def identifiers(tokens: List[JavaToken]) -> List[Identifier]:
    return [it for it in tokens if isinstance(it, Identifier)]

# Returns a list of Keyword tokens from the token list
def keywords(tokens: List[JavaToken]) -> List[Keyword]:
    return [it for it in tokens if isinstance(it, Keyword)]

# Returns a list of Literal tokens (strings, numbers, etc.)
def literals(tokens: List[JavaToken]) -> List[Literal]:
    return [it for it in tokens if isinstance(it, Literal)]

# Extracts all direct children of a node (recursively unpacks lists of children)
def children(node: Node) -> List:
    nodes = []

    for it in node.children:
        if isinstance(it, List):
            nodes += it
        else:
            nodes.append(it)

    return nodes

# Filters out empty lines from source code
def non_empty_lines(code: str) -> List[str]:
    return [line for line in code.split('\n') if line.strip() != '']

# Recursively collects all nodes of a specific type from the AST
def get_nodes(node, node_type) -> List:
    result = []

    if isinstance(node, node_type):
        result.append(node)

    for it in children(node):
        if isinstance(it, Node):
            result += get_nodes(it, node_type)

    return result

# Returns a count of all nodes of a given type in the AST
def get_nodes_count(node, node_type) -> int:
    return len(get_nodes(node, node_type))

# Maps a sorted set of values to unique integer IDs (e.g., for encoding)
def build_mapping_to_ids(values: Iterable) -> Dict:
    values = sorted(set(values))
    return {key: value for key, value in zip(values, range(len(values)))}
