# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from operator import mul
from math import log
import sys

class NGram:
 
    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        tokens = sents
        for sent in tokens:
            for i in range(n-1):
                sent.insert(i, '<s>')
            if sent[len(sent)-1] != '</s>':
                sent.append('</s>')
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                self.counts[ngram] += 1
                self.counts[ngram[:-1]] += 1
 
    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
 
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]
 
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
 
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        cond_prob = 0
        if self.counts[tuple(prev_tokens)] != 0:
            cond_prob = float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]
        return cond_prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        assert len(sent) > 0
        n = self.n

        test_sent = sent
        for i in range(n-1):
            test_sent.insert(i, '<s>')
        if test_sent[len(test_sent)-1] != '</s>':
            test_sent.append('</s>')

        for i in range(n-1, len(test_sent)):

            token = test_sent[i]
            prev_tokens = None

            if n > 1:
                prev_tokens = test_sent[i-n+1 : i]

            if i == n-1:
                sent_prob = self.cond_prob(token, prev_tokens)
            else:
                sent_prob *= self.cond_prob(token, prev_tokens)

        return sent_prob
 
    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        assert len(sent) > 0
        n = self.n

        test_sent = sent
        for i in range(n-1):
            test_sent.insert(i, '<s>')
        if test_sent[len(test_sent)-1] != '</s>':
            test_sent.append('</s>')

        log2 = lambda x: log(x, 2)

        sent_prob = 0

        for i in range(n-1, len(test_sent)):

            token = test_sent[i]
            prev_tokens = None

            if n > 1:
                prev_tokens = test_sent[i-n+1 : i]
            else:
                prev_tokens = []

            tokens = prev_tokens + [token]
            cond_prob = 0

            log_tks_count = float(self.counts[tuple(tokens)])
            if log_tks_count == 0.0:
                log_tks_count = float('-Inf')
            else:
                log_tks_count = log2(log_tks_count)

            log_prev_tks_count = float(self.counts[tuple(prev_tokens)])
            if log_prev_tks_count == 0.0:
                log_prev_tks_count = float('-Inf')
            else:
                log_prev_tks_count = log2(log_prev_tks_count)

            if log_tks_count == float('-Inf') and log_prev_tks_count == float('-Inf'):
                cond_prob = float('-Inf')
            else:
                cond_prob = log_tks_count - log_prev_tks_count

            sent_prob += cond_prob

        return sent_prob


class NGramGenerator:
 
    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.probs = defaultdict(dict)
        self.sorted_probs = defaultdict(list)

        for tokens, count in model.counts.items():
            if len(tokens) == model.n:
                prev_tokens = tokens[:-1]
                token = tokens[-1]
                self.probs[prev_tokens][token] = model.cond_prob(token, list(prev_tokens))
 
    def generate_sent(self):
        """Randomly generate a sentence."""
 
    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
 
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """


class AddOneNGram:
 
    """
       Todos los métodos de NGram.
    """
 
    def V(self):
        """Size of the vocabulary.
        """