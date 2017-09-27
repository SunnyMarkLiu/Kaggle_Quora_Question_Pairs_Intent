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
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
