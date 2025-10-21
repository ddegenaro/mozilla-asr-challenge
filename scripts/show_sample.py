import glob
import csv
import os
import argparse

ROOT = '/home/drd92/mozilla-asr-challenge/mcv-sps-st-09-2025'

corpus_tsvs = sorted(glob.glob(
    os.path.join(ROOT, '*', '*corpus*.tsv')
))

assert len(corpus_tsvs) == 21

def main(k: int):
    print('\n', '-' * 120, '\n', sep='')
    for i, corpus_tsv in enumerate(corpus_tsvs):
        lang = os.path.split(corpus_tsv)[1].split('-')[-1][:-4]
        if lang == 'CY':
            lang = 'el-CY'
        reader = csv.reader(open(corpus_tsv, 'r', encoding='utf-8'), delimiter='\t')
        next(reader) # skip header
        for _ in range(k):
            text = ''
            while not text:
                text = next(reader)[6].strip()
            print(i+1, ' ', lang, '\t', text, '\n', sep='')
        print('-' * 120, '\n', sep='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=1)
    args = parser.parse_args()
    main(args.k)