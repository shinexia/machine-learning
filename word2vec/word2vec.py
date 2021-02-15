import logging
import sys
import os
import argparse
import json
import time
import math
import warnings
import multiprocessing
import numpy as np

from typing import Dict, List, Iterator, Iterable

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s')

# copy from `word2vec.c`
_MAX_STRING = 100
_EXP_TABLE_SIZE = 1000
_MAX_EXP = 6
_MAX_SENTENCE_LENGTH = 1000
_MAX_CODE_LENGTH = 40
_VOCAB_HASH_SIZE = 30000000  # Maximum 30 * 0.7 = 21M words in the vocabulary

# other
_WORD_DELIMITERS = ' \t\n'
# Words occuring less than min_count times will be discarded from the vocab
_WORD_MIN_COUNT = 5
_MAX_VOCAB_SIZE = int(_VOCAB_HASH_SIZE * 0.7)
_EMBEDDINGS_DIM = 100
_END_OF_LINE = '</s>'


class Word(object):

    def __init__(self, word, count: int = 0, code: List[int] = [], point: List[int] = []):
        self.word = word
        self.count = count
        self.code = code  # Binary Huffman encoding from the leaf to the root
        self.point = point  # List of indices from the leaf to the root

    def __repr__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)

    def __str__(self):
        return self.__repr__()


class Vocab(object):

    def __init__(self, words=List[Word], total=None):
        self.words = words
        self.word_hash = {}
        self.size = len(self.words)
        self.total = total or len(self.words)

        for i, w in enumerate(self.words):
            self.word_hash[w.word] = i

    def __getitem__(self, idx):
        return self.words[idx]

    def search(self, word):
        return self.word_hash.get(word, -1)

    def __len__(self):
        return self.size


class Tokenizer(object):

    def __init__(self, train_file: str, offset=0, max_token_len=_MAX_STRING, delimiters=_WORD_DELIMITERS):
        self.train_file = train_file
        self.offset = offset
        self.max_token_len = max_token_len
        self.delimiters = delimiters

    def seek(self, offset):
        self.offset = offset

    def __call__(self) -> Iterator[str]:
        logging.info('reading tokens from: %s', self.train_file)
        with open(self.train_file, 'r', encoding='utf8') as f:
            if self.offset > 0:
                f.seek(self.offset)
            block = ''
            while True:
                a = f.read(4096)
                block += a
                if a == '':
                    # ignore the last token of the train_file
                    break
                while True:
                    skip = 0
                    for c in block:
                        if c in self.delimiters:
                            if c == '\n':
                                yield _END_OF_LINE
                            skip += 1
                        else:
                            break
                    block = block[skip:]
                    if block == '':
                        break
                    token = ''
                    for c in block:
                        if c in self.delimiters:
                            break
                        token += c
                    if len(token) == len(block):
                        # continue reading the next block
                        break
                    block = block[len(token):]
                    if token == '':
                        break
                    # keep the same behavior to the original implementation in `word2vec.c`
                    if len(token) > self.max_token_len-2:
                        token = token[:self.max_token_len-2]
                    yield token


class VocabBuilder(object):

    def __init__(self, max_vocab_size=_MAX_VOCAB_SIZE, min_count=_WORD_MIN_COUNT):
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.vocab_dict: Dict[str, Word] = dict()
        self.vocab_size = 0
        self.total_words = 0
        self.min_reduce = 1

    def build(self, token_sequences: Iterable[str]) -> Vocab:
        start_time = time.time()
        for word in token_sequences:
            self._add_word(word)
            if self.total_words % 10000 == 0:
                sys.stdout.write("\rreaded words %d" % self.total_words)
                sys.stdout.flush()
        sys.stdout.write("\rreaded words %d\n" % self.total_words)
        sys.stdout.flush()
        vocab = self._build_vocab()
        cost = time.time() - start_time
        logging.info('total words: %s, cost: %.3fs', self.total_words, cost)
        return vocab

    def _add_word(self, word: str):
        self.total_words += 1
        w = self.vocab_dict.get(word, None)
        if w is None:
            w = Word(word, count=1)
            self.vocab_dict[word] = w
            self.vocab_size += 1
        else:
            w.count += 1
        if self.vocab_size > self.max_vocab_size:
            self._reduce_vocab()

    def _reduce_vocab(self):
        for word in self.vocab_dict.values():
            if word.count <= self.min_reduce:
                self.vocab_dict.pop(word.word, None)
        self.min_reduce += 1

    def _build_vocab(self):
        words = list(self.vocab_dict.values())
        words.sort(key=lambda w: -w.count)
        keep = 0
        for w in words:
            if w.count < self.min_count:
                break
            keep += 1
        words = words[:keep]
        words = [Word(_END_OF_LINE, count=1)] + words
        vocab = Vocab(words=words, total=self.total_words)
        return vocab


class TreeNode(object):

    def __init__(self, word: Word = None, weight=None):
        self.word = word
        self.parent: int = -1
        self.weight = weight if weight is not None else word.count if word is not None else 0
        self.code = 0


class HuffmanTreeBuilder(object):

    def __init__(self, vocab: Vocab):
        # words must be sorted by count desc
        self.vocab = vocab
        self.vocab_words = vocab.words
        self.vocab_size = vocab.size

    def build(self) -> Vocab:
        logging.info('building huffman tree')
        nodes = self._build_tree()
        vocab_size = self.vocab_size
        for node in nodes[:vocab_size]:
            code = []
            point = []
            word = node.word
            while node.parent >= 0:
                code.append(node.code)
                point.append(node.parent)
                node = nodes[node.parent]
            code.reverse()
            point.reverse()
            word.code = code
            word.point = [p - vocab_size for p in point]
        return self.vocab

    def _build_tree(self) -> List[TreeNode]:
        vocab_size = self.vocab_size
        if vocab_size <= 0:
            return None, []
        nodes = [TreeNode(word=w) for w in self.vocab_words] + [TreeNode(weight=1e15) for _ in range(vocab_size - 1)]
        if vocab_size <= 1:
            return nodes
        pos1 = vocab_size - 1
        pos2 = vocab_size
        process_count = 1
        for i in range(vocab_size-1):
            if pos1 >= 0 and nodes[pos1].weight < nodes[pos2].weight:
                min1i = pos1
                pos1 -= 1
            else:
                min1i = pos2
                pos2 += 1
            if pos1 >= 0 and nodes[pos1].weight < nodes[pos2].weight:
                min2i = pos1
                pos1 -= 1
            else:
                min2i = pos2
                pos2 += 1
            node1i, node2i = nodes[min1i], nodes[min2i]
            parent = nodes[vocab_size + i]
            node1i.parent = vocab_size + i
            node2i.parent = vocab_size + i
            parent.weight = node1i.weight + node2i.weight
            node2i.code = 1
            print('min1i: %s=%s, min2i: %s=%s, parent: %s=%s' % (min1i, node1i.weight, min2i, node2i.weight, vocab_size + i, parent.weight))
            process_count += 1
            if process_count % 1000 == 0:
                sys.stdout.write("\rbuilded word %d" % process_count)
                sys.stdout.flush()
        sys.stdout.write("\rbuilded word %d\n" % process_count)
        sys.stdout.flush()
        return nodes


class Random(object):

    def __init__(self, seed=1):
        self.seed = np.uint64(seed)
        self.b = np.uint64(25214903917)
        self.c = np.uint64(11)

    def nextint(self, maxv=None):
        self.seed = self.seed * self.b + self.c
        if maxv is not None:
            return int(self.seed) % maxv
        return int(self.seed)


class DataLoader(object):

    def __init__(self, vocab: Vocab = None, tokenizer: Tokenizer = None, thread_id: int = 0, mode: str = 'cbow', window=2, batch_size: int = 1, max_sentence_len: int = _MAX_SENTENCE_LENGTH):
        assert mode in ['cbow', 'sg']
        self.thread_id = thread_id
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.mode = mode
        self.window = window
        self.batch_size = batch_size
        self.max_sentence_len = max_sentence_len

    def _single_generator(self, rand: Random = None):
        vocab = self.vocab
        word_sequences = self.tokenizer()
        sentence = []
        pos = 0
        rand = rand or Random(seed=self.thread_id)
        while True:
            if pos >= len(sentence):
                # read next sentence
                sentence = []
                word_count = 0
                while word_count < self.max_sentence_len:
                    word = next(word_sequences, None)
                    if word is None:
                        break
                    if word == _END_OF_LINE:
                        if word_count > 0:
                            break
                        else:
                            continue
                    word_idx = vocab.search(word)
                    if word_idx < 0:
                        continue
                    sentence.append(word_idx)
                    word_count += 1
                pos = 0
                if len(sentence) == 0:
                    # to the end
                    return
            center_word = sentence[pos]
            b = rand.nextint(self.window)
            window = self.window - b
            start_pos = max(0, pos - window)
            end_pos = min(pos + window + 1, len(sentence))
            context_words = sentence[start_pos:pos] + sentence[pos+1:end_pos]
            pos += 1
            if self.mode == 'cbow':
                yield context_words, center_word
            else:
                for word in context_words:
                    yield center_word, word

    def iter(self, rand: Random = None):
        single_generator = self._single_generator(rand=rand)
        if self.batch_size <= 1:
            return single_generator

        def batch_generator():
            xs, ys = [], []
            for x, y in single_generator:
                xs.append(x)
                ys.append(y)
                if len(xs) >= self.batch_size:
                    yield xs, ys
                    xs, ys = [], []
            # ignore the last one
        return batch_generator()

    def __iter__(self):
        return self.iter()


def create_exp_table(size=_EXP_TABLE_SIZE):
    table = [0.0] * size
    for i in range(size):
        v = (i / size * 2 - 1) * _MAX_EXP
        x = math.exp(v)
        table[i] = x / (x + 1)
        print('i: %d, v: %.8f, exp: %.8f' % (i, v, table[i]))
    return table


_exp_table = create_exp_table(size=_EXP_TABLE_SIZE)
_exp_factor = (int)(_EXP_TABLE_SIZE / (_MAX_EXP * 2))


def sigmoid(f):
    i = (int)((f + _MAX_EXP) * _exp_factor)
    return _exp_table[i]


class Model(object):

    def __init__(self, vocab: Vocab, embedding_dim: int = _EMBEDDINGS_DIM, mode: str = 'cbow', alpha=None, hs=1, negative=25):
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.alpha = alpha or 0.025
        self.hs = hs
        self.negative = negative

        self.syn0 = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float)
        self.syn1 = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float)

        rand = Random(seed=1)
        a = 0
        while a < self.vocab_size:
            b = 0
            while b < self.embedding_dim:
                next_random = rand.nextint()
                v = (((next_random & 0xffff) / 65536) - 0.5) / self.embedding_dim
                self.syn0[a][b] = v
                b += 1
            a += 1

        for i in range(self.vocab_size):
            print(('x: %d' % i) + ', syn0: ' + ','.join(['%.8f' % v for v in self.syn0[i]]))

        if self.mode == 'cbow':
            self.train = self._train_cbow
        else:
            self.train = self._train_skip_gram

    def _train_cbow(self, x, y):
        pass

    def _train_skip_gram(self, center_word, context_word):
        word_x = self.vocab[center_word]
        context_embedding = self.syn0[context_word]
        neu1e = np.zeros(self.embedding_dim)
        for d, point in enumerate(word_x.point):
            point_weight = self.syn1[point]
            f = np.dot(context_embedding, point_weight)
            if f >= _MAX_EXP:
                continue
            elif f <= -_MAX_EXP:
                continue
            else:
                z = sigmoid(f)
            g = (1 - word_x.code[d] - z) * self.alpha
            print('x: %s, y: %s, p: %s, f: %.8f, z: %.8f, g: %.8f' % (center_word, context_word, point, f, z, g))
            neu1e += g * point_weight
            self.syn1[point] += g * context_embedding
        self.syn0[context_word] += neu1e


def load_vocab(filepath: str) -> List[Word]:
    logging.info('reading vocab from: %s', filepath)
    with open(filepath, 'r', encoding='utf8') as f:
        words = []
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            spans = line.split(' ')
            word = Word(spans[0], count=int(spans[1]))
            words.append(word)
    return words


def save_vocab(vocab: Vocab, dest: str):
    logging.info('saving vocab to: %s', dest)
    with open(dest, 'w+') as f:
        for word in vocab.words:
            f.write(word.word)
            f.write(' ')
            f.write(str(word.count))
            f.write(' ')
            f.write(str(len(word.code)))
            f.write(' ')
            f.write(','.join([str(c) for c in word.code]))
            f.write(' ')
            f.write(','.join([str(c) for c in word.point]))
            f.write('\n')


def save_vector(vector, vocab: Vocab, dest: str):
    logging.info('saving vector to: %s', dest)
    with open(dest, 'w+') as f:
        f.write(str(len(vector)))
        f.write(' ')
        f.write(str(len(vector[0])))
        f.write('\n')
        for i, v in enumerate(vector):
            f.write(vocab[i].word)
            for j, x in enumerate(v):
                f.write(' %.8f' % x)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str, default='../data/text8', help='Use text data from <file> to train the model')
    parser.add_argument('-output', type=str, default=0, help='Use <file> to save the resulting word vectors / word clusters')
    parser.add_argument('-size', type=int, default=_EMBEDDINGS_DIM, help='size of word vectors; default is 100')
    parser.add_argument('-window', type=int, default=5, help='Set max skip length between words; default is 5')
    parser.add_argument('-sample', type=float, default=1e-3, help='Set threshold for occurrence of words. Those that appear with higher frequency in the training data\n'
                        'will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)')
    parser.add_argument('-hs', type=int, default=0, help='Use Hierarchical Softmax; default is 0 (not used)')
    parser.add_argument('-negative', type=int, default=5, help='Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)')
    parser.add_argument('-threads', type=int, default=multiprocessing.cpu_count(), help='Use <int> threads (default cpu_count)')
    parser.add_argument('-iter', type=int, default=5, help='Run more training iterations (default 5)')
    parser.add_argument('-min-count', type=int, default=_WORD_MIN_COUNT, help='This will discard words that appear less than <int> times; default is 5')
    parser.add_argument('-alpha', type=float, default=None, help='Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW')
    parser.add_argument('-classes', type=int, default=0, help='Output word classes rather than word vectors; default number of classes is 0 (vectors are written)')
    parser.add_argument('-debug', type=int, default=2, help='Set the debug mode (default = 2 = more info during training)')
    parser.add_argument('-binary', type=int, default=0, help='Save the resulting vectors in binary moded; default is 0 (off)')
    parser.add_argument('-save-vocab', type=str, default='logs/vocab.txt', help='The vocabulary will be saved to <file>')
    parser.add_argument('-read-vocab', type=str, default='logs/vocab.txt', help='The vocabulary will be read from <file>, not constructed from the training data')
    parser.add_argument('-cbow', type=int, default=1, help='Use the continuous bag of words model; default is 1 (use 0 for skip-gram model')

    parser.add_argument('-save-token', type=str, default='logs/token.txt', help='The tokenized words will be saved to <file>')
    parser.add_argument('-batch-size', '-bs', type=int, default=1, help='batch size')
    args = parser.parse_args()
    logging.info('args: %s', args)

    tokenizer = Tokenizer(train_file=args.train)
    vocab = VocabBuilder(min_count=args.min_count).build(tokenizer())
    vocab = HuffmanTreeBuilder(vocab).build()

    print('Starting training using file %s' % args.train)
    print('Vocab size: %s' % vocab.size)
    print('Words in train file: %s' % vocab.total)

    save_vocab(vocab, dest=args.save_vocab)

    mode = 'cbow' if args.cbow == 1 else 'sg'
    dataloader = DataLoader(vocab=vocab, tokenizer=tokenizer, mode=mode, batch_size=args.batch_size, window=args.window)
    model = Model(vocab, embedding_dim=args.size, mode=mode, alpha=args.alpha)

    train_rand = Random(seed=0)
    for epoch in range(args.iter):
        print('epoch: %s' % epoch)
        for x, y in dataloader.iter(rand=train_rand):
            model.train(x, y)

    save_vector(model.syn0, vocab=vocab, dest=args.output)


if __name__ == '__main__':
    main()
