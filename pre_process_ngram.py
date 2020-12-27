import json
import argparse
import pickle as pkl
from collections import defaultdict


def precook(s, n=4):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  # lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def build_dict(vids, wtoi):
    refs_words = []
    refs_idxs = []
    count_vids = 0
    for vid in vids:
        ref_words = []
        ref_idxs = []
        for cap in vids[vid]['final_captions']:
            tmp_tokens = cap
            tmp_tokens = [_ if _ in wtoi else '<UNK>' for _ in tmp_tokens]
            ref_words.append(' '.join(tmp_tokens))
            ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
        refs_words.append(ref_words)
        refs_idxs.append(ref_idxs)
        count_vids += 1
    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_vids


def main():
    vids = json.load(open('captions.json'))
    wtoi = json.load(open('info.json'))['word_to_idx']

    ngram_words, ngram_idxs, ref_len = build_dict(vids, wtoi)

    pkl.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(
        'output.pkl'+'-words.p', 'wb'))
    pkl.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(
        'output.pkl'+'-idxs.p', 'wb'))

main()