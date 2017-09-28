#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-27 上午10:52
"""
from __future__ import absolute_import, division, print_function

import re

punctuations = {
    ".": " ",
    ",": " ",
    ";": " ",
    ":": " ",
    "\"": "",
    "\'": "",
    "!": "",
    "?": "",
    "(": " ",
    ")": " ",
    "[": " ",
    "]": " ",
    "=": " ",
    "/": " ",
    "#": " ",
    "@": " ",
}

whitespaces = {
    "\n+": " ",
    "\t+": " ",
}

shorthands = {
    "&amp": "and",
    "w/": "with",
    "w/o": "without",
    "b/c": "because",
    "b/t": "between",
    " b4 ": " before ",
    " 2day": " today",
}

numbers = {
    "1": " one ",
    "2": " two ",
    "3": " three ",
    "4": " four ",
    "5": " five ",
    "6": " six ",
    "7": " seven ",
    "8": " eight ",
    "9": " nine ",
    "0": " zero ",
}

ordinals = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
    "11th": "eleventh",
    "12th": "twelfth",
}

acronyms = [
    (re.compile(r"^US\s|\sUS\s|\sUS$"), " United States "),
    (re.compile(r"^EU\s|\sEU\s|\sEU$"), " European Union "),
]

unicodes = {
    u"\u2018": "'",  # left single-quote
    u"\u2019": "'",  # right single-quote
    u"\u201A": ",",  # weird low single-quote/comma
    u"\u201B": "'",  # another left single-quote

    u"\u0027": "'",  # apostrophe
    u"\u05F3": "'",  # hebrew punct. "geresh"
    u"\uFF07": "'",  # full-width apostrophe

    u"\u201C": '"',  # left double-quote
    u"\u201D": '"',  # right double-quote
    u"\u201F": '"',  # weird left double-quote

    u"\uFF06": "and",  # full ampersand
    u"\u0026": "and",  # normal ampersand
    u"\uFE60": "and",  # small ampersand

    u"\u2026": "...",  # ellipsis
    u"\uFF0D": "-",  # full-width hyphen
    u"\u2010": "-",  # hyphen
    u"\u2011": "-",  # hyphen
    u"\u2012": "-",  # figure-dash
}

smileys = {
    ":‑)": "feeling happy",
    ":)": "feeling happy",
    ":D": "feeling happy",
    ":o)": "feeling happy",
    ":]": "feeling happy",
    ":3": "feeling happy",
    ":c)": "feeling happy",
    ":>": "feeling happy",
    "=]": "feeling happy",
    "8)": "feeling happy",
    "=)": "feeling happy",
    ":}": "feeling happy",
    ":^)": "feeling happy",
    ":‑D": "feeling happy",
    "8‑D": "feeling happy",
    "8D": "feeling happy",
    "x‑D": "feeling happy",
    "xD": "feeling happy",
    "X‑D": "feeling happy",
    "XD": "feeling happy",
    "=‑D": "feeling happy",
    "=D": "feeling happy",
    "=‑3": "feeling happy",
    "=3": "feeling happy",
    "B^D": "feeling happy",
    ":-))": "feeling happy",
    ":'‑)": "feeling happy",
    ":')": "feeling happy",
    ":*": "feeling happy",
    ":-*": "feeling happy",
    ":^*": "feeling happy",
    "(": "feeling happy",
    "'}{'": "feeling happy",
    ")": "feeling happy",
    ">:[": "feeling sad",
    ":‑(": "feeling sad",
    ":(": "feeling sad",
    ":‑c": "feeling sad",
    ":c": "feeling sad",
    ":‑<": "feeling sad",
    ":っC": "feeling sad",
    ":<": "feeling sad",
    ":‑[": "feeling sad",
    ":[": "feeling sad",
    ":{": "feeling sad",
    ";(": "feeling sad",
    ":-||": "feeling sad",
    ":@": "feeling sad",
    ">:(": "feeling sad",
    ":'‑(": "feeling sad",
    ":'(": "feeling sad",
    "D:<": "feeling sad",
    "D:": "feeling sad",
    "D8": "feeling sad",
    "D;": "feeling sad",
    "D=": "feeling sad",
    "DX": "feeling sad",
    "v.v": "feeling sad",
    "D‑':": "feeling sad",
    ">:\\": "feeling sad",
    ">:/": "feeling sad",
    ":‑/": "feeling sad",
    ":‑.": "feeling sad",
    ":/": "feeling sad",
    ":\\": "feeling sad",
    "=/": "feeling sad",
    "=\\": "feeling sad",
    ":L": "feeling sad",
    "=L": "feeling sad",
    ":S": "feeling sad",
    ">.<": "feeling sad",
    ":|": "feeling sad",
    ":‑|": "feeling sad",
}

contractions = {

    r"ain\'t": "is not",
    r"can\'t": "cannot",
    r"shan\'t": "shall not",
    r"sha\'n't": "shall not",
    r"won\'t": "will not",
    r"let\'s": "let us",
    r"how\'d": "how did",
    r"how\'d'y": "how do you",
    r"where'd": "where did",

    r"'m": " am ",
    r"'d": " would had ",
    r"n\'t": " not ",
    r"\'ve": " have ",
    r"\'re": " are ",
    r"\'ll": " will ",
    r"\'s": " is ",

    r"\'cause": "because",
    r"ma'am": "madam",
    r"o'clock": "of the clock",
    r"y'all": "you all",
}
