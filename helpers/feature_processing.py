from typing import Dict
from typing import List

import javalang
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .layout import *
from .lexical import *
from .syntax import MaxDepthASTNode
from .syntax import ASTNodeBigramsTF
from .syntax import ASTNodeTypesTF
from .syntax import JavaKeywords

from .utils import build_mapping_to_ids

def get_features_helper(code: str) -> Dict:
    """
    Calculates a set of features for the given source code.

    :return: dictionary with features
    """

    file_length = len(code)
    if file_length == 0:
        print(code)
    tokens = list(javalang.tokenizer.tokenize(code))
    tree = javalang.parse.parse(code)

    features = {}

    # LEXICAL FEATURES
    features.update(calculate_WordUnigramTF(tokens))
    features.update(calculate_NumKeyword(tokens, file_length))
    features.update(calculate_NumTokens(tokens, file_length))
    features.update(calculate_NumLiterals(tokens, file_length))
    features.update(calculate_NumKeywords(tokens, file_length))
    features.update(calculate_NumFunctions(tree, file_length))
    features.update(calculate_NumComments(code))
    features.update(calculate_NumTernary(tree, file_length))
    features.update(calculate_AvgLineLength(code))
    features.update(calculate_StdDevLineLength(code))
    features.update(calculate_AvgParams(tree))
    features.update(calculate_StdDevNumParams(tree))

    # LAYOUT FEATURES
    features.update(calculate_NumTabs(code))
    features.update(calculate_NumSpaces(code))
    features.update(calculate_NumEmptyLines(code))
    features.update(calculate_WhiteSpaceRatio(code))
    features.update(calculate_NewLineBeforeOpenBrace(code))
    features.update(calculate_TabsLeadLines(code))

    # SYNTAX FEATURES
    features.update(MaxDepthASTNode.calculate(tree))
    features.update(ASTNodeBigramsTF.calculate(tree))
    features.update(ASTNodeTypesTF.calculate(tree))
    features.update(JavaKeywords.calculate(tokens))

    return features

def process_code(code_snippets: tuple) -> Dict:
    user_id, code_snippet, index_id = code_snippets
    try:
        return (user_id, get_features_helper(code_snippet), index_id)
    except javalang.parser.JavaSyntaxError as e:
        # Log the Java syntax error.
        print("[ERR]", e.description, "[FILE]", user_id, "[AT]", e.at)
        return None  # Return None to indicate failure
    except Exception as e:
        # Log any other exceptions that may occur.
        print("[ERR] An unexpected error occurred while processing", user_id)
        print(e)
        return None

def get_features_for_snippets(code_snippets: List[str], n_jobs: int = -1) -> List[Dict]:
    """
    Calculates sets of features for the given code snippets.

    :param files: list of file paths
    :param n_jobs: number of parallel jobs
    :return: list with features for each code snippet
    """

    # Parallel computing just to make it faster.
    with Parallel(n_jobs=n_jobs) as pool:
        features = pool(delayed(process_code)(snippet) for snippet in code_snippets)

    # Filter out any None results (snippets that had errors).
    features = [feature for feature in features if feature is not None]

    return features


def build_sample(sample: Dict, feature_to_id: Dict) -> np.array:
    features = np.empty(len(feature_to_id))
    features[:] = np.nan

    for key, value in sample.items():
        index = feature_to_id[key]
        features[index] = value

    return features


def build_dataset(samples: List[Dict], n_jobs: int = -1) -> pd.DataFrame:
    """
    Builds a pandas data frame from the given list of feature sets.

    :param samples: list of features
    :param n_jobs: number of jobs
    :return: data frame with all features
    """

    feature_names = set()

    for sample in samples:
        feature_names |= sample.keys()

    feature_names = sorted(feature_names)
    feature_to_id = build_mapping_to_ids(feature_names)

    with Parallel(n_jobs=n_jobs) as pool:
        features = pool(delayed(build_sample)(sample, feature_to_id)
                        for sample in samples)

    features = pd.DataFrame(features)

    print("[CREATED FEATURES DF]")
    features.columns = feature_names

    return features