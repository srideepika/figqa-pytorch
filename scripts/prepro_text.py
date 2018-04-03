#!/usr/bin/env python
import os
import os.path as pth
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from PIL import Image
from figqa.utils import datasets
from figqa.utils.datasets import ques_to_tensor

def tokenize_qas(qa_pairs):
    for qa_pair in qa_pairs:
        qa_pair['question'] = word_tokenize(qa_pair['question'])
        if 'UNK' in qa_pair['question']:
            print(qa_pair['question'], qa_pair['question'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Process figureqa text into tensors: '
                'Extract vocab, answer tensor, question tensor')

    # Input files
    parser.add_argument('--figqa-dir',
        default='data/SHAPES/questions/',
        help='directory containing unzipped figqa files')
    # NOTE: These should be processed at once because the complete
    # set determines the vocab.
#    splits = ['no_annot_test1', 'no_annot_test2', 'validation1',
 #             'validation2', 'train1', 'sample_train1']
    splits = ['shapes_train','shapes_val','shapes_test']
    # Output files
    parser.add_argument('--output-dir',
        default='data/SHAPES_pre/',
        help='Save one hdf5 per dataset here')
    # Options
    parser.add_argument('--max_ques_len', default=40, type=int,
        help='Max length of questions')
    args = parser.parse_args()

    # 1: load QA pairs and extract vocab
    qa_pairs = {}
    vocab = set()
    for split in splits:
        with open(pth.join(args.figqa_dir, split, 'qa_pairs.json'), 'r') as f:
            #print(f)
            #print(json.load(f))
            split_qa_pairs = json.load(f)
        tokenize_qas(split_qa_pairs)
        qa_pairs[split] = split_qa_pairs
        for qap in split_qa_pairs:
            vocab.update(qap['question'])
    vocab = sorted(list(vocab))
    vocab = ['NULL', '<START>', '<END>'] + vocab

    # 2: save vocab
    print('output',args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    vocab_fname = pth.join(args.output_dir, 'vocab.json')
    word2ind = {w: i for i, w in enumerate(vocab)}
    with open(vocab_fname, 'w') as f:   
        json.dump({
            'ind2word': vocab, # just a list
            'word2ind': word2ind
        }, f)

    # 3: save questions and answers as tensors (without start and end tokens)
    for split in splits:
        os.makedirs(pth.join(args.output_dir, split), exist_ok=True)
        fname = pth.join(args.output_dir, split, 'qa_pairs.h5')
        print('filename',fname)
        f = h5py.File(fname, 'w')

        # questions -> tensor
        questions = (qap['question'] for qap in qa_pairs[split])
        questions = [ques_to_tensor(q, word2ind) for q in questions]
        questions = np.stack(questions)
        f.create_dataset('questions', data=questions)
        # answers -> tensor (0 or 1 for no or yes)
#        if split not in ['no_annot_test1', 'no_annot_test2']:
        d={}
        d['false']=0
        d['true']=1
        d['']=-1
        if split not in ['shapes_test']:
            answers = [d[qap['answer']] for qap in qa_pairs[split]]
            print(answers)
            answers = np.array(answers, dtype='uint32')
            f.create_dataset('answers', dtype='uint32', data=answers)
        # image indices
        image_idx = [qap['image_filename'][7:-4] for qap in qa_pairs[split]]
        print([(qap['image_filename'],qap['image_filename'][7:-4]) for qap in qa_pairs[split]])
        print( image_idx[0])
        image_idx = np.array(image_idx,dtype='uint32')
        f.create_dataset('image_idx',dtype='uint32', data=image_idx)

        f.close()

