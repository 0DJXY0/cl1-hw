# CMSC 723
# Template by: Jordan Boyd-Graber
# Homework submission by: NAME

import sys
from typing import Iterable, Tuple, Dict


import nltk
from nltk.corpus import dependency_treebank
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.classify.api import ClassifierI
from nltk.parse.dependencygraph import DependencyGraph

from nltk.corpus import dependency_treebank
nltk.download('dependency_treebank')
nltk.download('universal_tagset')
kROOT = 'ROOT'
VALID_TYPES = set(['s', 'l', 'r'])


def flatten(xss: Iterable) -> Iterable:
    """
    Flatten a list of list into a list
    """
    return [x for xs in xss for x in xs]

def split_data(transition_sequence, proportion_test=10,
               generate_test=False, limit=-1):
    """
    Iterate over stntences in the NLTK dependency parsed corpus and create a
    feature representation of the sentence.

    :param proportion_test: 1 in every how many sentences will be test data?
    :param test: Return test data only if C{test=True}.
    :param limit: Only consider the first C{limit} sentences
    """
    for ii, ss in enumerate(dependency_treebank.parsed_sents()):
        item_is_test = (ii % proportion_test == 0)

        if limit > 0 and ii > limit:
            break

        example = {"sentence": ss, "features": []}
        for ff in transition_sequence(ss):
            example["features"].append(ff.feature_representation())

        if item_is_test and generate_test:
            yield example
        elif not generate_test and not item_is_test:
            yield example

class Transition:
    """
    Class to represent a dependency graph transition.
    """
    
    def __init__(self, type, edge=None):
        self.type = type
        self.edge = edge
        self.features = {}

        assert self.type in VALID_TYPES

    def __str__(self):
        return "Transition(%s, %s)" % (self.type, str(self.edge))
        
    def add_feature(self, feature_name: str, feature_value: float):
        """
        Add a feature to the transition.

        :param feature_name: The name of the feature
        :param feature_value: The value of the feature
        """

        self.features[feature_name] = feature_value

    def feature_representation(self) -> Tuple[Dict[str, float], str]:
        """
        Create a training instance for a classifier: the classifier will predict 
        the transition type from the features you create.
        """

        return (self.features, self.type)

    def pretty_print(self, sentence):
        """
        Pretty print the transition that is a part of the sentence.

        :param sentence: The sentence that the transition is a part of
        """

        if self.edge:
            a, b = self.edge
            return "%s\t(%s, %s)" % (self.type,
                                     sentence.get_by_address(a)['word'],
                                     sentence.get_by_address(b)['word'])
        else:
            return self.type
        
class ShiftReduceState:
    """
    Class to represent the state of the shift-reduce parser.
    """

    def __init__(self, words: Iterable[str], pos: Iterable[str]):
        """
        :param words: A list of words
        :param pos: A list of POS tags
        """

        assert words[0] == kROOT, "First word must be ROOT"
        assert len(words) == len(pos), "Words and POS tags must be the same length"

        self.words = words
        self.pos = pos

        self.edges = []

        self.stack = [0]
        self.buffer = list(range(1, len(words)))
        self.buffer.reverse()

    def pretty_print(self):
        return "Stack: %s\n Buffer: %s\n Edges: %s" % (str(self.stack), str(self.buffer), str(self.edges))

    def apply(self, transition: Transition):
        
        if transition.type == 's':
            self.shift()
        elif transition.type == 'l':
            self.left_arc()
        elif transition.type == 'r':
            self.right_arc()

    def shift(self) -> Transition:
        """
        Shift the top of the buffer to the stack and return the transition.
        """

        index = -1
        assert len(self.buffer) > 0, "Buffer is empty for shift"
        # Implement this
        self.stack.append(self.buffer[index])
        self.buffer.pop(index)

        return Transition('s', None)

    def left_arc(self) -> Transition:
        """
        Create a new left dependency edge and return the transition.
        """

        stack_top = -1

        assert len(self.buffer) > 0, "Buffer is empty for left arc"
        assert len(self.stack) > 0, "Stack is empty for left arc"

        # Implement this
        assert self.stack[-1] > 0, "The top of the stack is 0"
        buffer_top = self.buffer[-1]
        stack_top = self.stack[-1]
        self.edges.append((buffer_top, stack_top))
        self.stack.pop(-1)
        
        return Transition('l', (buffer_top, stack_top))

    def right_arc(self) -> Transition:
        """
        Create a new right dependency edge and return the transition.
        """

        stack_top = -1

        assert len(self.buffer) > 0, "Buffer is empty for right arc"
        assert len(self.stack) > 0, "Stack is empty for right arc"

        # Implement this
        stack_top = self.stack.pop(-1)
        buffer_top = self.buffer.pop(-1)
        self.edges.append((stack_top, buffer_top))
        self.buffer.append(stack_top)
        return Transition('r', (stack_top, buffer_top))
    
    def feature_extractor(self, index: int) -> Iterable[Tuple[str, float]]:
        """
        Given the state of the shift-reduce parser, create a feature vector from the
        sentence.

        :param index: The current offset of the word under consideration (wrt the
        original sentence).

        :return: Yield tuples of feature -> value
        """
        
        yield ("Buffer size", len(self.buffer))
        yield ("Stack size", len(self.stack))
        yield ("Part of the speech", index)
    
        # Implement this
        if len(self.stack) > 0:
            stop = self.stack[-1]
            stop_word = self.words[stop]
            stop_pos = self.pos[stop]
            yield("Top of the stack",stop_word)
            yield("Top of the stack pos",stop_pos)
        else:
            yield ("Top of the stack", "None")
            yield ("Top of the stack pos", "None")
        
        if len(self.buffer) >= 2:
            b1 = self.buffer[-1]
            b2 = self.buffer[-2]
            b1_word = self.words[b1]
            b1_pos = self.pos[b1]
            b2_word = self.words[b2]
            b2_pos = self.pos[b2]
            yield("Top of the buffer",b1_word)
            yield("Top of the buffer pos",b1_pos)
            yield("Top two of the buffer",b1_word + ' ' + b2_word)
            yield("Top two of the buffer pos",b1_pos + ' ' + b2_pos)
        else:
            yield("Top of the buffer","None")
            yield("Top of the buffer pos","None")
            yield ("Top two of the buffer", "None")
            yield ("Top two of the buffer pos", "None")
        
        left_most_verb = None 
        for i in range(1, len(self.words)):
            if self.pos[i] == 'VERB':
                left_most_verb = self.words[i]
                break
        if left_most_verb is not None:
            yield ("Left most verb", left_most_verb) 
        else:
            yield ("Left most verb", "None")  



def heuristic_transition_sequence(sentence: DependencyGraph) -> Iterable[Transition]:
    """
    Implement this for extra credit
    """
    universal_rules = {
        ('VERB', 'VERB'),
        ('VERB', 'NOUN'),
        ('VERB', 'ADV'),
        ('VERB', 'ADP'),
        ('VERB', 'CONJ'),
        ('VERB', 'DET'),
        ('VERB', 'NUM'),
        ('VERB', 'ADJ'),
        ('VERB', 'X'),
        ('NOUN', 'ADJ'),
        ('NOUN', 'DET'),
        ('NOUN', 'NUM'),
        ('NOUN', 'NOUN'),
        ('ADP', 'NOUN'),
        ('ADP', 'ADV'),
        ('ADJ', 'ADV'),
    }


    words = []
    pos = []

    for i in range(len(sentence.nodes)):
        node = sentence.nodes[i]
        if i == 0:
            words.append(kROOT)
            pos.append('ROOT')
        else:
            words.append(node['word'])
            pos.append(node['tag'])

    num_words = len(words)
    parents = {}
    children = dict([(k, []) for k in range(num_words)])
    # find the left-most verb 
    left_most_verb = None
    for i in range(1, num_words):
        if pos[i] == 'VERB':
            left_most_verb = i
            break

    if left_most_verb is not None:
        parents[left_most_verb] = 0
        children[0].append(left_most_verb)

    for i in range(1, num_words):
        if i in parents:
            continue  # already has a parent

        possible_heads = []
        min_distance = num_words  

        # Search for the nearest word w' satisfies the rules
        for j in range(1, num_words):
            if i == j:
                continue
            if (pos[j], pos[i]) in universal_rules:
                distance = abs(i - j)
                if distance < min_distance:
                    min_distance = distance
                    possible_heads = [j]
                elif distance == min_distance:
                    possible_heads.append(j)

        if possible_heads:
            head = min(possible_heads)
            parents[i] = head
            children[head].append(i)
        else:
            parents[i] = 0
            children[0].append(i)
    # print(words)
    # print('heads:',heads)
    # # Build nodes dictionary
    # nodes = {}
    # for i in range(num_words):
    #     nodes[i] = {'address': i, 'word': words[i], 'tag': pos[i], 'head': heads[i]}

    # # Create a DependencyGraph
    # dg = DependencyGraph()
    # dg.nodes = nodes
    # return list(transition_sequence(dg))
    parents[0] = None
    # print(parents)
    # print(children)
    transitions = []
    sr = ShiftReduceState(words,pos)
    while len(sr.buffer) >0 or len(sr.stack) > 1:
        # print(sr.stack)
        # print(sr.buffer)
        if len(sr.stack) >= 1:
            # print(children)
            l = sr.stack[-1]
            r = sr.buffer[-1]
            if parents[l] == r:
                children[r].remove(l)
                transitions.append(sr.left_arc())
            elif parents[r] == l and len(children[r]) == 0:
                if r in children[l]:
                    children[l].remove(r)
                transitions.append(sr.right_arc())
            else:
                transitions.append(sr.shift())
        
        #if only root is in the stack
        else:
            transitions.append(sr.shift())


    return transitions

 

def classifier_transition_sequence(classifier: MaxentClassifier, sentence: DependencyGraph) -> Iterable[Transition]:
    """
    Unlike transition_sequence, which uses the gold stack and buffer states,
    this will predict given the state as you run the classifier forward.

    :param sentence: A dependency parse tree

    :return: A list of transition objects that reconstructs the depndency
    parse tree as predicted by the classifier.
    """

    # Complete this for extra credit
    words=[]
    pos = []
    # print(sentence.nodes.items())
    # pos=[node.get("tag", "") for _, node in sorted(sentence.nodes.items())]
    for _, node in sentence.nodes.items():
        words.append(node["word"])
        pos.append(node["tag"])
    #     print(node['deps'][''])
    # print(words)
    # print(pos)
    words[0] = kROOT
    sr = ShiftReduceState(words,pos)

    transitions = []
    while len(sr.buffer) >0 or len(sr.stack) > 1:
        if len(sr.buffer) == 1 and len(sr.stack) == 0 and 0 in sr.buffer:
            transitions.append(sr.shift())
            break
        index = sr.buffer[-1] if len(sr.buffer) > 0 else -1
        feature_sets = dict(sr.feature_extractor(index))
        action = classifier.classify(feature_sets)
        prob = classifier.prob_classify(feature_sets)
        # prob_dict = {}
        # prob_dict['s'] = prob.prob('s')
        # prob_dict['l'] = prob.prob('l')
        # prob_dict['r'] = prob.prob('r')
        # order = []
        # for key, value in prob_dict.items():
        #     print(key)
        #     print(value)
        action = prob.generate()
        t = None
        # print('buffer: ',sr.buffer)
        # print('stack: ',sr.stack)
        if action == 's':
            if len(sr.buffer) > 1:
                t = sr.shift()
            else:
                if len(sr.stack) >= 1:
                    t = sr.right_arc()
                else:
                    t = sr.left_arc()
        elif action == 'l':
            if len(sr.stack) >= 1 and sr.stack[-1]>0:
                t = sr.left_arc()
            else:
                t = sr.shift() if len(sr.buffer) > 1 else sr.right_arc()                 

        elif action == 'r':
            if len(sr.stack) >= 1:
                t = sr.right_arc()
            else:
                t = sr.shift() if len(sr.buffer) > 1 else sr.left_arc()  
        else:
            t = sr.shift()
        if t != None:
            transitions.append(t)
    return transitions
        
def transition_sequence(sentence: DependencyGraph) -> Iterable[Transition]:
    """
    :param sentence: A dependency parse tree

    :return: A list of transition objects that creates the dependency parse
    tree.
    """
    words=[]
    pos = []
    # print(sentence.nodes.items())
    # pos=[node.get("tag", "") for _, node in sorted(sentence.nodes.items())]
    for _, node in sorted(sentence.nodes.items()):
        words.append(node["word"])
        pos.append(node["tag"])
    #     print(node['deps'][''])
    # print(words)
    # print(pos)
    words[0] = kROOT
    sr = ShiftReduceState(words,pos)
    num_words = len(sentence.nodes)
    parents = {}
    children = {}
    for i in range(num_words):
        node = sentence.nodes[i]
        parents[i] = node['head']
        children[i] = node['deps']['']
    # print('parents',parents)
    # print(children)
    while len(sr.buffer) >0 or len(sr.stack) > 1:
        if len(sr.stack) >= 1:
            # print(children)
            l = sr.stack[-1]
            r = sr.buffer[-1]
            if parents[l] == r:
                children[r].remove(l)
                transition = sr.left_arc()
            elif parents[r] == l and len(children[r]) == 0:
                if r in children[l]:
                    children[l].remove(r)
                transition = sr.right_arc()
            else:
                transition = sr.shift()
        
        #if only root is in the stack
        else:
            transition = sr.shift()

        index = sr.buffer[-1] if len(sr.buffer) > 0 else -1
        for feat_name, feat_value in sr.feature_extractor(index):
            transition.add_feature(feat_name, feat_value)
        # print('stack:',sr.stack)
        # print('buffer:',sr.buffer)
        yield transition


    # for node_id, node in sorted(sentence.nodes.items()):
    #     print(node_id)
    #     if node_id == 0:
    #         continue
    #     dep = node['deps']['']
    #     print(dep)
    #     # while sr.buffer:
    #     #     if 
    #     # print(node)
    # print(sr.pretty_print())
    


 
    

    # Exclude the root node





    return
    yield # We write this yield to make the function iterable

def parse_from_transition(word_sequence: Iterable[Tuple[str, str]], transitions: Iterable[Transition]) -> DependencyGraph:
  """
  :param word_sequence: A list of tagged words (list of pairs of words and their POS tags) that
  need to be built into a tree

  :param transitions: A a list of Transition objects that will build a tree.

  :return: A dependency parse tree
  """
  assert len(transitions) >= len(word_sequence), "Not enough transitions"

  # insert root if needed
  if word_sequence[0][0] != kROOT:
    word_sequence.insert(0, (kROOT, 'TOP'))
  sent = ['']*(len(word_sequence))         
#   print(word_sequence)
  words = [w for w, _ in word_sequence]
  pos = [t for _, t in word_sequence]

  sr = ShiftReduceState(words, pos)
  for t in transitions:
      sr.apply(t)
  
  # get the head index of each word
#   print('edges:',sr.edges)
  for i in range(len(words)):
      word = words[i]
      tag = pos[i]
      head = None
      for h, d in sr.edges:
          if d == i:
              head = h
              break
      if head is None and i != 0:
          head = 0  
      elif head is None and i == 0:
          head = -1  # Head of root which need to be deleted
      line = f"{word}\t{tag}\t{head}"
      sent[i] = line
      
#   sent[0] = f"{'ROOT'}\t{'TOP'}\t{-1}"
  # You're allowed to create your DependencyGraph however you like, but this
  # is how I did it.
  reconstructed = '\n'.join(sent[1:])
#   print('sent',sent)
#   print('recons:',reconstructed)
#   print('Graph: ',nltk.parse.dependencygraph.DependencyGraph(reconstructed))
  return nltk.parse.dependencygraph.DependencyGraph(reconstructed)

def sentence_attachment_accuracy(reference: DependencyGraph, sample: DependencyGraph) -> float :
    """
    Given two parse tree transition sequences, compute the number of correct
    attachments (ROOT is always correct)
    """

    correct = 0   
    for idx in reference.nodes:
        if idx == 0:
            continue
        ref_head = reference.nodes[idx]['head']
        sample_head = sample.nodes[idx]['head']
        if ref_head == sample_head:
            correct += 1                                                 
    return correct

def attachment_accuracy(classifier: ClassifierI, reference_sentences: Iterable[DependencyGraph]) -> float:
    """
    Compute the average attachment accuracy for a classifier on a corpus
    of sentences.
    """
    correct = 0
    num_attachments = 0
    
    for i in reference_sentences:
        # Implement this for extra credit
        reference = i["sentence"]
        words=[]
        for _, node in sorted(reference.nodes.items()):
            words.append((node["word"],node["tag"]))
        words[0] = (kROOT,'TOP')
        transitions = classifier_transition_sequence(classifier,reference)
        sample = parse_from_transition(words,transitions)
        
        correct += sentence_attachment_accuracy(reference, sample)
        # print('correct:',sentence_attachment_accuracy(reference, sample))
        # print('total1:',len(words))
        num_attachments += len(words) - 1                                     

    return correct / num_attachments

def classifier_accuracy(classifier: ClassifierI, reference_transitions: Iterable[Tuple[Dict[str, float], str]]) -> float:
    """
    Compute the average attachment accuracy for a classifier on a corpus
    of sentences.

    """

    correct = 0
    total = 0
    for sentence in reference_transitions:
        predictions = [classifier.classify(x[0]) for x in sentence["features"]]
        labels = [x[1] for x in sentence["features"]]
        temp = sum(1 for x, y in zip(predictions, labels) if x == y)
        correct += temp
        total += len(predictions)
    return correct / total

def heuristic_attachment_accuracy(reference_sentences: Iterable[DependencyGraph]) -> float:
    """
    Compute the average attachment accuracy for the heuristic parser on a corpus
    of sentences.
    """
    correct = 0
    num_attachments = 0
    
    for i in reference_sentences:
        # Implement this for extra credit
        reference = i["sentence"]
        words=[]
        for _, node in sorted(reference.nodes.items()):
            words.append((node["word"],node["tag"]))
        words[0] = (kROOT,'TOP')
        transitions = heuristic_transition_sequence(reference)
        sample = parse_from_transition(words,transitions)
        
        correct += sentence_attachment_accuracy(reference, sample)
        num_attachments += len(words) - 1                                     

    return correct / num_attachments


if __name__ == "__main__":
    from itertools import chain
    # Create an example sentence
    from test_dependency import kCORRECT
    sent = nltk.parse.dependencygraph.DependencyGraph(kCORRECT)
    words = [x.split('\t')[0] for x in kCORRECT.split('\n')]
    words = [kROOT] + words
    # To delete
    # print(sent)
    # w=[]
    # for _, node in sorted(sent.nodes.items()):
    #     w.append((node["word"],node["tag"]))
    # w[0] = (kROOT,'TOP')
    # print(w)
    #####
    # trans = []
    for ii in transition_sequence(sent):
        print(ii.pretty_print(sent))
        # trans.append(ii)
    
    # parse_from_transition(w,trans)
    
    train_data = list(split_data(transition_sequence))
    test_data = list(split_data(transition_sequence, generate_test=True))

    print("Training data: %i" % len(train_data))
    print("Test data: %i" % len(test_data))

    feature_examples = []
    for sentence in train_data:
        feature_examples += sentence["features"]
    classifier = MaxentClassifier.train(feature_examples, algorithm='IIS', max_iter=25, min_lldelta=0.001)

    # Classification accuracy
    classifier_acc = classifier_accuracy(classifier, test_data)
    print('Held-out Classification Accuracy: %6.4f' % classifier_acc)

    attachment_acc = attachment_accuracy(classifier, test_data)
    print('Held-out Attachment Accuracy:     %6.4f' % attachment_acc)

    heuristic_acc = heuristic_attachment_accuracy(test_data)
    print('Heuristic Attachment Accuracy:    %6.4f' % heuristic_acc)

