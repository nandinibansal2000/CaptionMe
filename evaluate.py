import json
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from utils import train_test_split_, greedySearch
import numpy as np
import keras
import os
from Caption_Preprocessing import CaptionPreprocessing
import pickle


def compute_rouge(original, predicted):
    '''
    original: list of list of captions
    [
      ['this is a dog', 'this is a puppy'],
      ['this is a cat', 'this is a kitten'],
    ]

    predicted: ['this is a sleepy dog',
                'the kitten is playing'
                ]
    '''

    hyps, refs = [], []
    for j in range(len(predicted)):
        hyps += [i for i in original[j]]
        refs += [predicted[j]]*len(original[j])
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    print(scores)


def get_bleu_score(original, predicted):
    '''
    original: list of list of captions
    [
      ['this is a dog', 'this is a puppy'],
      ['this is a cat', 'this is a kitten'],
    ]

    predicted: ['this is a sleepy dog',
                'the kitten is playing'
                ]
    '''

    candidates = [can.split() for can in predicted]
    references = []
    for cap_set in original:
        cap_set_tokenized = [can.split() for can in cap_set]
        references.append(cap_set_tokenized)

    ind_1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    ind_2 = corpus_bleu(references, candidates, weights=(0, 1, 0, 0))
    ind_3 = corpus_bleu(references, candidates, weights=(0, 0, 1, 0))
    ind_4 = corpus_bleu(references, candidates, weights=(0, 0, 0, 1))

    cum_2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    cum_3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
    cum_4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))

    gleu_1 = corpus_gleu(references, candidates,min_len=1, max_len=1)
    gleu_2 = corpus_gleu(references, candidates,min_len=1, max_len=2)
    gleu_3 = corpus_gleu(references, candidates,min_len=1, max_len=3)
    # print(ind_1, ind_2, ind_3, ind_4)
    print('cumulative 1,2,3,4 bleu', ind_1, cum_2, cum_3, cum_4)
    # print(gleu_1, gleu_2, gleu_3)


def get_pred(test_images, test_captions, maxLen, word_to_idx, idx_to_word, model):
    test_y = []
    predicted_y = []
    for i in range(len(test_images)):
        if i%100 == 0:
            print(i)
        test_img = test_images[i][1]
        test_cap = test_captions[i][1]
        test_cap = [i.replace('startseq ', '').replace(' endseq', '') for i in test_cap]
        test_y.append(test_cap)
        photo = test_img.reshape(test_img.shape[0], )
        predicted_cap = greedySearch(photo, maxLen, word_to_idx, idx_to_word, model)
        predicted_y.append(predicted_cap)

    return test_y, predicted_y


def evaluate():
    image_features_path = './Dataset/Saved_Json/all_image_features_30k.pkl'
    caption_path = './Dataset/Saved_Json/dataset.csv'
    saved_json_path = './Dataset/Saved_Json'
    model_path = os.path.join(saved_json_path, 'model_final.h5')
    img_to_image_features = pickle.load(open(image_features_path, 'rb'))
    glove_path = './Dataset/GloVe/glove.6B.200d.txt'

    captionPreprocess = CaptionPreprocessing(caption_path, saved_json_path, glove_path)
    captionPreprocess.preprocess()
    img_to_caption = captionPreprocess.image_to_caption
    vocab_size = len(captionPreprocess.vocab)
    word_to_idx = captionPreprocess.word_to_idx
    idx_to_word = captionPreprocess.idx_to_word
    embdgs_map = captionPreprocess.word_embd_map
    maxLen = captionPreprocess.maxLen

    print(maxLen)

    train_images, test_images, train_captions, test_captions = train_test_split_(img_to_caption, img_to_image_features, test_size=0.3)

    model = keras.models.load_model(model_path)
    print('Model loaded successfully...')
    test_y, predicted_y = get_pred(test_images, test_captions, maxLen, word_to_idx, idx_to_word, model)

    get_bleu_score(test_y, predicted_y)
    compute_rouge(test_y, predicted_y)


if __name__ == '__main__':
    evaluate()