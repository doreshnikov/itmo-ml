import os


def read_all():
    return [read_from_dir(f'lab04_bayes/resources/part{i}') for i in range(1, 11)]


def read_from_dir(dirname):
    files = [[], []]
    for filename in os.listdir(dirname):
        with open(f'{dirname}/{filename}') as f:
            subject = f.readline()[9:].split()
            f.readline()
            text = f.readline().split()
            cls = 0 if 'legit' in filename else 1
            files[cls].append((subject, text))
    return files


def ngrams(text, n=2, sep=' '):
    return [sep.join(text[i:i + n]) for i in range(len(text) - n)]


def collect_ngrams(header, text, n=2):
    return set(ngrams(header, n, '-') + ngrams(text, n, '+'))
