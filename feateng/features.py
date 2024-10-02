# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """

        question -- The JSON object of the original question, you can extract metadata from this such as the category

        run -- The subset of the question that the guesser made a guess on

        guess -- The guess created by the guesser

        guess_history -- Previous guesses (needs to be enabled via command line argument)

        other_guesses -- All guesses for this run
        """


        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?

        guess_length = 0
        guess_length = log(1 + len(guess))

        # How many words long is the question?


        # How many characters long is the guess?
        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", guess_length)  

class LengthNormal(Feature):
   def __init__(self, name):
      self.name = name
      self.max_len = 0

   def add_training(self, question_source):                                
      import json                                                         
      import gzip                                                         
      if 'json.gz' in question_source:                                    
          with gzip.open(question_source) as infile:                      
              questions = json.load(infile)                               
      else:                                                               
          with open(question_source) as infile:                           
              questions = json.load(infile)                               
      for ii in questions:                                                
          self.counts[self.normalize(ii["page"])] += 1   

class LengthRun(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?

        run_length = 0
        run_length = log(1 + len(run))

        # How many words long is the question?


        # How many characters long is the guess?
        if run is None or run=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", run_length)  
            
class FrequencyFeature(Feature):
   def __init__(self, name):
      from eval import normalize_answer
      self.name = name
      self.counts = Counter()
      self.normalize = normalize_answer

   def add_training(self, question_source):                                
      import json                                                         
      import gzip                                                         
      if 'json.gz' in question_source:                                    
          with gzip.open(question_source) as infile:                      
              questions = json.load(infile)                               
      else:                                                               
          with open(question_source) as infile:                           
              questions = json.load(infile)                               
      for ii in questions:                                                
          self.counts[self.normalize(ii["page"])] += 1                    

   def __call__(self, question, run, guess, guess_history, other_guesses=None): 
    #   print('question: ',question)
    #   print('run: ',run)
    #   print('guess: ', guess)
    #   print('guess_history',guess_history)
    #   print('other guesses: ',other_guesses)           
      yield ("guess", log(1 + self.counts[self.normalize(guess)]))        

class UselessInfo(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # print('question: ',question)
        # How many characters long is the question?

        num_useless = 0
        useless = ('a','the')
        for word in run.split():
            if word.lower() in useless:
                num_useless += 1

        # num_useless = log(1+num_useless)
        # How many words long is the question?


        # How many characters long is the guess?
        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", num_useless)   
        
class GenderInfo(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?

        num_gender = 0
        gender = ('his','her','him','she','he','male','female','it','its','their','them','theirs')
        for word in run.split():
            # print(word.lower())
            if word.lower() in gender:
                num_gender += 1

        # num_useless = log(1+num_useless)
        # How many words long is the question?


        # How many characters long is the guess?
        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", num_gender)           

class QuestionCategory(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    def __init__(self,name):
        self.name = name
        self._category = dict()
        self.idx = 0

    def add_training(self, question_source):                                
        import json                                                         
        import gzip                                                         
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)                               
        for ii in questions:
            if ii['category'] not in self._category:
                self.idx += 1
                self._category[ii['category']] = self.idx
                                              
    def __call__(self, question, run, guess, guess_history, other_guesses=None):

        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", self._category[question['category']])    

class Year(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):

        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", int(question['year'])/2024)          

class Difficulty(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    # def __init__(self,name):
    #     self.name = name
    #     self._difficulty = dict()
    #     self.idx = 0

    # def add_training(self, question_source):                                
    #     import json                                                         
    #     import gzip                                                         
    #     if 'json.gz' in question_source:                                    
    #         with gzip.open(question_source) as infile:                      
    #             questions = json.load(infile)                               
    #     else:                                                               
    #         with open(question_source) as infile:                           
    #             questions = json.load(infile)                               
    #     for ii in questions:
    #         if ii['difficulty'] not in self._difficulty:
    #             self.idx += 1
    #             self._difficulty[ii['difficulty']] = self.idx
                                              
    # def __call__(self, question, run, guess, guess_history, other_guesses=None):

    #     if guess is None or guess=="":  
    #         yield ("guess", -1)         
    #     else:                           
    #         yield ("guess", self._difficulty[question['difficulty']])        
    def __call__(self, question, run, guess, guess_history, other_guesses=None):

        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", question['difficulty'])   

class Tournament(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    def __init__(self,name):
        self.name = name
        self._tournament = dict()
        self.idx = 0

    def add_training(self, question_source):                                
        import json                                                         
        import gzip                                                         
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)                               
        for ii in questions:
            if ii['tournament'] not in self._tournament:
                self.idx += 1
                self._tournament[ii['tournament']] = self.idx
                                              
    def __call__(self, question, run, guess, guess_history, other_guesses=None):

        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", self._tournament[question['tournament']])    


class Prompt(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """
    def __init__(self,name):
        self.name = name
        self._prompt = dict()
        self.idx = 0

    def add_training(self, question_source):                                
        import json                                                         
        import gzip                                                         
        if 'json.gz' in question_source:                                    
            with gzip.open(question_source) as infile:                      
                questions = json.load(infile)                               
        else:                                                               
            with open(question_source) as infile:                           
                questions = json.load(infile)                               
        for ii in questions:
            if ii['answer_prompt'] not in self._prompt:
                self.idx += 1
                self._prompt[ii['answer_prompt']] = self.idx
                                              
    def __call__(self, question, run, guess, guess_history, other_guesses=None):

        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:                           
            yield ("guess", self._prompt[question['answer_prompt']])  

class GamePlay(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):

        if guess is None or guess=="":  
            yield ("guess", -1)         
        else:  
            if question['gameplay'] == True:                         
                yield ("guess", 1)  
            else:
                yield ("guess", 0) 
if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from parameters import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags, guesser_params)
    buzzer = load_buzzer(flags, buzzer_params)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
