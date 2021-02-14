# word2vec

re-implementing  word2vec

## Build

### Preparing

```bash
mkdir -p logs
```

### word2vec.py

```bash
python word2vec.py -train text8.min -output logs/vectors.bin -save-vocab logs/vocab.txt -cbow 0 -size 10 -window 2 -negative 0 -hs 1 -sample 0 -threads 1 -binary 0 -min-count 3 -iter 1 > logs/a.txt
```

### ref (modified)

```bash
make
./word2vec -train text8.min -output logs/vectors.ref.bin -save-vocab logs/vocab.ref.txt -cbow 0 -size 10 -window 2 -negative 0 -hs 1 -sample 0 -threads 1 -binary 0 -min-count 3 -iter 1  > logs/b.txt
```

### diff

```bash
vimdiff logs/vectors.bin logs/vectors.ref.bin
vimdiff logs/a.txt logs/b.txt
```
