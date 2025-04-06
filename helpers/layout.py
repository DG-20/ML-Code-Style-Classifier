from typing import Dict
from .utils import non_empty_lines


def calculate_NumTabs(code: str) -> Dict:
    value = code.count('\t') / len(code)
    return {'ln(numTabs/length)': value}


def calculate_NumSpaces(code: str) -> Dict:
    value = code.count(' ') / len(code)
    return {'ln(numSpaces/length)': value}


def calculate_NumEmptyLines(code: str) -> Dict:
    lines = code.strip().split('\n')
    value = sum(map(lambda it: it == '', lines)) / len(code)
    return {'ln(numEmptyLines/length)': value}


def calculate_WhiteSpaceRatio(code: str) -> Dict:
    num_space = sum(map(lambda it: it.isspace(), code))
    value = num_space / (len(code) - num_space)
    return {'whiteSpaceRatio': value}


def calculate_NewLineBeforeOpenBrace(code: str) -> Dict:
    lines = code.split('\n')
    new_line_cnt = sum('{' == line.strip() for line in lines)
    total_cnt = sum('{' in line for line in lines)
    value = 1 if 2 * new_line_cnt > total_cnt else 0
    return {'newLineBeforeOpenBrace': value}


def calculate_TabsLeadLines(code: str) -> Dict:
    lines = non_empty_lines(code)
    space_cnt = sum(line[0] == ' ' for line in lines)
    tab_cnt = sum(line[0] == '\t' for line in lines)
    value = 1 if tab_cnt > space_cnt else 0
    return {'tabsLeadLines': value}
