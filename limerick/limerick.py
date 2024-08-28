# Author: YOUR NAME HERE
# Date: DATE SUBMITTED

# Use word_tokenize to split raw text into words
import nltk
nltk.download('cmudict')
import json

from nltk.tokenize import word_tokenize
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
        list = word.split(' ')
        print(list)
        count = 0
        for w in list:
            try:
                print(self._pronunciations[w.lower()])
                syllables = self._pronunciations[w.lower()]
                shorter = min(map(len,syllables))
                count += shorter
            except KeyError:
                count += 1
        return count
    
    def after_stressed(self, word):
        """
        For each of the prounciations, yield whatever is after the
        last stressed syllable.  If there are no stressed syllables,
        return the whole proununciation.
        """

        # TODO: Complete this function

        pronunciations = self._pronunciations.get(self._normalize(word), [])
        print('ps: ',pronunciations)
        for pronunciation in pronunciations:
            print(pronunciation)
            s = self.stress(pronunciation)
            ls = [i for i, e in enumerate(s) if e != 0]
            last = ls[-1]
            if last == len(s):
                yield pronunciation
            else:
                for i in range(last+1,len(s)):
                    yield pronunciation[i]

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



        return False

    def last_words(self, lines):
        """
        Given a list of lines in a list, return the last word in each line
        """
        # TODO: Complete this function
        return None

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

        # TODO: Complete this function



        return False

if __name__ == "__main__":
    ld = LimerickDetector()

    limerick_tests = ld.load_json("sample_limericks.json")
    
    words = ["billow", "pillow", "top", "America", "doghouse", "two words", "Laptop", "asdfasd"]
    for word in words:
        ld.after_stressed(word)
        print('stress printed')
    #     print(word)
    #     print(ld.num_syllables(word))
    # print('end')
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
