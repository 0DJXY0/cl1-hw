# Author: YOUR NAME HERE
# Date: DATE SUBMITTED

# Use word_tokenize to split raw text into words
import nltk
# nltk.download('cmudict')
# nltk.download('punkt_tab')
import json

from nltk.tokenize import word_tokenize
import string
from string import punctuation, digits
import re

class LimerickDetector:
    @staticmethod
    def load_json(filename):
        with open("sample_limericks.json") as infile:
            data = json.load(infile)

        limericks = []
        for example in data:
            limericks.append((example["limerick"], "\n".join(example["lines"])))

        return limericks
        
    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()
        self._vowels = lambda x: [y for y in x if y[-1] in digits]

    def _normalize(self, a):
        """
        Do reasonable normalization so we can still look up words in the
        dictionary
        """

        return a.lower().strip()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.  
        """

        # TODO: Complete this function

        

        try:
            # print(self._pronunciations[word.lower()])
            count_list =[]
            syllables = self._pronunciations[word.lower()]
            for i in syllables:
                count = 0
                for j in i:
                    if j[-1].isdigit():
                        count += 1
                count_list.append(count)
            # if len(count_list) >=2 and count_list[0]!=count_list[1]:
            #     print(word)
            #     raise KeyboardInterrupt
            return min(count_list)
            # return min(map(len,syllables))
                
        except KeyError:
            return 1
        
    
    def after_stressed(self, word):
        """
        For each of the prounciations, yield whatever is after the
        last stressed syllable.  If there are no stressed syllables,
        return the whole proununciation.
        """

        # TODO: Complete this function

        pronunciations = self._pronunciations.get(self._normalize(word), [])
        if pronunciations == []:
            yield [word]
        # print('ps: ',pronunciations)
        for pronunciation in pronunciations:
            s = self.stress(pronunciation)
            ls = [i for i, e in enumerate(s) if e != 0]
            last = ls[-1]
            if last == len(s):
                yield pronunciation
            else:
                yield pronunciation[last:len(s)]

    def stress(self,pron):
        s = [0 for _ in range(len(pron))]
        for i in range(len(pron)):
            phone = pron[i]
            for char in phone:
                if char.isdigit():
                    s[i] = int(char)
        return s
    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.

        We use the definition from Wikipedia:

        Given two pronuncation lookups, see if they rhyme.  We use the definition from Wikipedia:

        A rhyme is a repetition of the exact phonemes in the final
        stressed syllables and any following syllables of words.

        """
        # TODO: Complete this function
        # Look up the pronunciations and get the prounciation after
        # the stressed vowel
        p_a = list(self.after_stressed(a))
        p_b = list(self.after_stressed(b))
        for i in p_a:        
            for j in p_b:
                if i == j:
                    return True


        return False

    def last_words(self, lines):
        """
        Given a list of lines in a list, return the last word in each line
        """
        # TODO: Complete this function
        last = []
        for line in lines:
            tmp = line.strip(punctuation)
            tmp = self.apostrophe_tokenize(tmp)
            # table = string.maketrans("","")
            # tmp = tmp.translate(table, punctuation)
            last.append(tmp[-1])
        # print(last)
        return last

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (and possibly the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """

        text = text.strip()
        lines = text.split('\n')
        # print(lines)
        # TODO: Complete this function
        l = len(lines)
        if l != 5:
            return False
        last = self.last_words(lines)
        if not self.rhymes(last[0],last[1]):
            # print('0 1: ',self.rhymes(last[0],last[1]))
            return False
        if not self.rhymes(last[0],last[4]):
            # print('0 4: ',self.rhymes(last[0],last[4]))
            return False
        if not self.rhymes(last[1],last[4]):
            # print('1 4: ',self.rhymes(last[1],last[4]))
            return False   
        if not self.rhymes(last[2],last[3]):
            # print('2 3: ',self.rhymes(last[2],last[3]))
            return False



        return True
    def apostrophe_tokenize(self, text):
        text = text.replace("'", "")
        return word_tokenize(text)

if __name__ == "__main__":
    ld = LimerickDetector()

    limerick_tests = ld.load_json("sample_limericks.json")
    
    words = ["billow", "pillow", "top", "America", "doghouse", "two words", "Laptop", "asdfasd","ab"]

    # for key, values in ld._pronunciations.items():
    #     ld.num_syllables(key)
    
    for display, func in [["Syllables", ld.num_syllables],
                          ["After Stressed", lambda x: list(ld.after_stressed(x))],
                          ["Rhymes", lambda x: "\t".join("%10s%6s" % (y, ld.rhymes(x, y)) for y in words)]]:
        print("=========\n".join(['', "%s\n" % display, '']))
        for word in words:
            print("%15s\t%s" % (word, str(func(word))))

    print(limerick_tests)
    for result, limerick in limerick_tests:
        print("=========\n")
        print(limerick)
        print("Truth: %s\tResult: %s" % (result, ld.is_limerick(limerick)))
