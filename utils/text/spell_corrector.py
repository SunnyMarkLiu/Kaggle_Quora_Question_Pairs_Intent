#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
A simple spell checker and corrector which can correct the word error.
TODO: Use linguistic model to correct the sentence error.
REF: https://github.com/junlulocky/SpellCorrector

@author: MarkLiu
@time  : 17-9-27 下午4:07
"""
from __future__ import absolute_import, division, print_function

from time import time
import re, collections


class EnglishSpellCorrector(object):
    def __init__(self, corpus):
        start = time()
        print("load spell corpus from {}, start time: {}".format(corpus, start))
        self.dictionary = self.__langModel(
            self.__get_words(file(corpus).read())
        )  # all the words in the language model
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        stop = time()
        print("corpus loaded, stop time: {}, cost {}s".format(stop, str(stop - start)))

    def __get_words(self, text):
        """
        Converts every word to lowercase
        :param text: input word sequence
        :return: separated word sequence in lowercase
        """
        return re.findall('[a-z]+', text.lower())

    def __langModel(self, wordseq):
        """
        Language Model: Using work frequency to get the probability of the word
        :param wordseq: input big word sequence
        :return: the frequency of each word in the word sequence
        """

        # specify the default value of a key to be 1, i.e., the smoothing in Language Model
        # if the word is not in the dictionary, the value will be 1.
        wordCount = collections.defaultdict(lambda: 1)
        for word in wordseq:
            wordCount[word] += 1
        return wordCount

    def __dist1_words(self, word):
        """
        Assuming the word length is n, see below the number of each error type.
        Get all the possible similar words which has edit distance of 1 compared the input word
        :param word: input word
        :return: all the possible similar words which has edit distance of 1 compared the input word
        """

        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]  # n deletions
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]  # n-1 transpositions
        replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]  # 26n alterations
        inserts = [a + c + b for a, b in splits for c in self.alphabet]  # 26(n+1) insertions
        return set(deletes + transposes + replaces + inserts)

    def __dist2_words(self, word):
        """
        Get all the possible similar words which has edit distance of 2 compared the input word
        and which is in the language model.
        :param word: input word
        :return: all the possible similar words which has edit distance of 2 compared the input word
        """
        return set(word2 for word1 in self.__dist1_words(word) for word2 in self.__dist1_words(word1))

    def __legal_words(self, words):
        """
        Get all the words in the dictionary.
        :param words: input word sequence
        :return: words in the dictionary
        """
        return set(w for w in words if w in self.dictionary)

    def correct_word(self, word):
        """
        Correct one word
        :param word: input word
        :return: correct word
        """
        # treat the distance 1 error and distance 2 error as equal probability
        # the main idea to put the last candidates of [words] is that we treat novel word having frequency 1
        possibleWords = self.__legal_words([word]) or \
                        self.__legal_words(self.__dist1_words(word)) or \
                        self.__legal_words(self.__dist2_words(word)) or [word]

        return max(possibleWords, key=self.dictionary.get)

    def correct_sentences(self, sentence):
        """
        Correct word sequence
        :param sentence: input word sequence
        :return: correct word sequence
        """
        words = self.__get_words(sentence)

        # return set(correct_word(word) for word in words )
        return ' '.join(self.correct_word(word) for word in words)
