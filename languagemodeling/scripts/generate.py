"""
Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from languagemodeling.ngram import NGramGenerator

if __name__ == '__main__':
    opts = docopt(__doc__)

    filename = opts['-i']

    train_set = open(filename, 'rb')
    ngram_gen = pickle.load(train_set)
    input_file.close()

    no_sents = int(opts['-n'])

    for i in range(no_sents):
        sent = ngram_gen.generate_sent()
        print(sent + ' ')