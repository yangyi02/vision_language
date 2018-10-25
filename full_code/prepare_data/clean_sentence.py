# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:30:17 2015

@author: yangyi05
"""

import re

def process_english_punctuation(text_in):
    """
    Process punctuation for English sentence
    """
    # Punctuations such as , and ' will not be removed
    punct = [';', r"/", '[', ']', '"', '{', '}', ':',
             '(', ')', '=', '+', '\\', '_', '-',
             '>', '<', '@', '`', '?', '!', '.', '*']
    comma_strip = re.compile(r"(\d)(\,)(\d)")
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")

    text_out = text_in
    for pun in punct:
        if (pun + ' ' in text_in or ' ' + pun in text_in) or \
                (re.search(comma_strip, text_in) != None):
            text_out = text_out.replace(pun, '')
        else:
            text_out = text_out.replace(pun, ' ')
    text_out = text_out.replace(',', ' , ')
    text_out = period_strip.sub("", text_out, re.UNICODE)
    return text_out

def process_chinese_punctuation(text_in):
    """
    Process punctuation for Chinese sentence
    """
    punct = ['；', '“', '”', '‘', '’', '【', '】', '『', '』', '–',
             '（', '）', '=', '+', '——', '-', '、', '：', '⊙', '≡', '￣', '﹏',
             '》', '《', '，', '？', '！', '～', '。', '●', '≥', '≤', '·',
             '…', '~', '^', '⊙ ', 'ω', '▽', '→', '←', '↓', '↑']

    text_out = text_in
    for pun in punct:
        text_out = text_out.replace(pun, ' ')

    punct = [';', r"/", '[', ']', '"', '{', '}', ':',
             '(', ')', '=', '+', '\\', '_', '-', "'",
             '>', '<', '@', '`', '?', '!', '.', '*']
    for pun in punct:
        text_out = text_out.replace(pun, ' ')

    return text_out

def process_english_digit_article(text_in):
    """
    Process digital articles for English sentence
    """
    # articles = ['a', 'an', 'the']
    # manual_map = {'none': '0',
    #              'zero': '0',
    #              'one': '1',
    #              'two': '2',
    #              'three': '3',
    #              'four': '4',
    #              'five': '5',
    #              'six': '6',
    #              'seven': '7',
    #              'eight': '8',
    #              'nine': '9',
    #              'ten': '10'}
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
        "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
        "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
        "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
        "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
        "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
        "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
        "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
        "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
        "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
        "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
        "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
        "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
        "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
        "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
        "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
        "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
        "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
        "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
        "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
        "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
        "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
        "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
        "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
        "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
        "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
        "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
        "youre": "you're", "youve": "you've"
    }

    text_out = []
    text_out = text_in.lower().split()
    # for word in temp_text:
    #    word = manual_map.setdefault(word, word)
    #    if word not in articles:
    #        text_out.append(word)
    for word_id, word in enumerate(text_out):
        if word in contractions:
            text_out[word_id] = contractions[word]
    return ' '.join(text_out)

def process_chinese_digit_article(text_in):
    """
    Process digital articles for Chinese sentence
    """
    # manual_map = {
    #    '0': '零',
    #    '1': '一',
    #    '2': '二',
    #    '3': '三',
    #    '4': '四',
    #    '5': '五',
    #    '6': '六',
    #    '7': '七',
    #    '8': '八',
    #    '9': '九',
    #    '10': '十'
    # }

    text_out = text_in.lower().split()
    # for word in temp_text:
    #    word = manual_map.setdefault(word, word)
    #    text_out.append(word)
    # return ' '.join(text_out)
    return text_out

def process_blank(text_in):
    """
    Process blanks for sentence, works both in English and Chinese
    """
    text_out = text_in.replace('\n', ' ').strip()
    text_out = text_out.split(' ')
    text_out = [word for word in text_out if len(word) > 0]
    text_out = ' '.join(text_out)
    return text_out

def process_english_sentence(text_in):
    """
    Process English sentences
    """
    text_out = process_blank(text_in)
    text_out = process_english_punctuation(text_out)
    text_out = process_english_digit_article(text_out)
    text_out = process_blank(text_out)
    return text_out

def process_chinese_sentence(text_in):
    """
    Process Chinese sentences
    """
    text_out = process_blank(text_in)
    text_out = process_chinese_punctuation(text_out)
    text_out = process_chinese_digit_article(text_out)
    text_out = process_blank(text_out)
    return text_out
