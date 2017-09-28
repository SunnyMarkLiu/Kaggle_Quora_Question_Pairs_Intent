#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-26 下午9:12
"""
from __future__ import absolute_import, division, print_function

import os
import re

from nltk.tokenize import WordPunctTokenizer

from . import acronyms_pattern as acronyms

current_path = os.path.abspath(__file__)
current_path = current_path[:-16]


class TextPreProcessor(object):
    def __init__(self, spell_correct=False,
                 spell_corpus_path=current_path+'/spellcheck_wikipedia_cleaned_corpus.txt'):
        self.spell_corrector = None
        if spell_correct:
            from .spell_corrector import EnglishSpellCorrector
            self.spell_corrector = EnglishSpellCorrector(spell_corpus_path)

    def clean_unit(self, text):
        """
        单位文本的清洗
        :param text: the string of text
        """
        text = re.sub(r"(\d+)kgs\s*", lambda m: m.group(1) + ' kilogram ', text)  # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg\s*", lambda m: m.group(1) + ' kilogram ', text)  # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k\s*", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)
        return text

    def clean_contractions(self, text):
        """
        缩略词的清洗, 返回完整的词汇
        :return: 
        """
        CON = acronyms.contractions
        for contraction in CON:
            text = re.sub(contraction, CON[contraction], text)
        return text

    def remove_link_text(self, text):
        """
        去除链接
        """
        text = re.sub(r"(\S*)https?://\S*", lambda m: m.group(1), text)
        return text

    def translate_acronyms(self, text):
        """
        常用缩略语的翻译, 如 "US" ==> "United States"
        """
        # acronym_trans is a (re, replacement) tuple
        ACR = acronyms.acronyms
        for ac in ACR:
            text = ac[0].sub(ac[1], text)
        return text

    def translate_emoji(self, text):
        """
        表情 emoji 的翻译 ":‑)" ==> "feeling happy"
        """
        SMY = acronyms.smileys
        for smy in SMY:
            text = text.replace(smy, ' ' + SMY[smy] + ' ')
        return text

    def translate_unicode(self, text):
        """
        unicode 字符翻译, 如 u"\u2018" ==> "'"
        :param text: 
        :return: 
        """
        UCD = acronyms.unicodes
        for u in UCD:
            text = text.replace(u, UCD[u])
        return text

    def translate_punctuation(self, text):
        """
        标点符号的清洗
        """
        PUN = acronyms.punctuations
        for p in PUN:
            text = text.replace(p, PUN[p])
        return text

    def translate_whitespace(self, text):
        """
        去除空白字符
        """
        WTS = acronyms.whitespaces
        for w in WTS:
            text = text.replace(w, WTS[w])
        return text

    def translate_shorthand(self, text):
        """
        翻译常用简写词, 如 "2day" ==> " today"
        """
        STH = acronyms.shorthands
        for s in STH:
            text = text.replace(s, STH[s])
        return text

    def translate_numbers_simple(self, text):
        """
        数字的翻译, "1" ==> "one"
        """
        NUM = acronyms.numbers
        for key, rep in NUM.items():
            text = text.replace(key, rep)
        return text

    def translate_ordinals(self, text):
        """
        顺序数字的翻译, 如 "1st" ==> "first"
        """
        ORD = acronyms.ordinals
        for key, rep in ORD.items():
            text = re.sub(" {0}".format(key), " {0}".format(rep), text)
            text = re.sub("^{0} ".format(key), "{0} ".format(rep), text)
            text = re.sub(" {0}$".format(key), " {0}".format(rep), text)
        return text

    def load_english_stopwords(self, filename):
        with open(filename, 'r') as f:
            return [line.rstrip('\n') for line in f.readlines()]

    def clean_text(self, text,
                   lower_case=True,
                   filter_stopwords=True,
                   keep_negative_words=False,
                   own_stopwords_file=None):
        """
        Clean text 
        :param text: the string of text
        :param lower_case: cast lower case
        :param filter_stopwords: filter english stop words
        :param keep_negative_words: filter english stop words, keep words like 'no, not'
        :param own_stopwords_file: yourown stopwords file
        :return: text string after cleaning
        """
        # basic cleaning
        text = self.clean_unit(text)
        text = self.clean_contractions(text)
        text = self.remove_link_text(text)
        text = self.translate_acronyms(text)
        text = self.translate_emoji(text)
        text = self.translate_unicode(text)
        text = self.translate_punctuation(text)
        text = self.translate_whitespace(text)
        text = self.translate_shorthand(text)
        text = self.translate_ordinals(text)

        # clean space and filter stop words
        text = text.lower() if lower_case else text
        if filter_stopwords:
            if own_stopwords_file:
                english_stopwords = self.load_english_stopwords(own_stopwords_file)
            elif not keep_negative_words:
                english_stopwords = self.load_english_stopwords(current_path+'/english_stopwords.vocab')
            else:
                english_stopwords = self.load_english_stopwords(current_path+'/english_stopwords_no_negative_words.vocab')

            word_tokenize = WordPunctTokenizer().tokenize
            text = [word for word in word_tokenize(text) if word not in english_stopwords]
            text = ' '.join(text)

        if self.spell_corrector:
            text = self.spell_corrector.correct_sentences(text)

            # using spelling_corrections.json to spell correct

        return text


if __name__ == "__main__":
    preprocessor = TextPreProcessor(spell_correct=True,
                                    spell_corpus_path=current_path+'/spellcheck_wikipedia_cleaned_corpus.txt')
    print(preprocessor.clean_text('teis is a simpl spel corrector'))
