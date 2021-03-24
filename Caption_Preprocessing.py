import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import string


class CaptionPreprocessing:
    def __init__(self, caption_path, saved_json_path, datasetSize='30k', glove_path=""):
        self.saved_json_path = saved_json_path
        self.caption_path = caption_path
        self.glove_path = glove_path
        self.datasetSize = datasetSize
        self.image_to_caption = None
        self.maxLen = None
        self.vocab = None
        self.word_to_idx, self.idx_to_word = None, None
        self.word_embd_map = None

    def preprocess(self):
        saved_json_files = [
            path for path in os.listdir(self.saved_json_path) if path.endswith(".json")
        ]

        if self.datasetSize == '8k':
            self.image_to_caption = self.create_img_to_captions(self.caption_path)
        else:
            self.image_to_caption = self.create_img_to_cap(self.caption_path)

        self.maxLen = self.get_maxlen_caption(self.image_to_caption)
        self.vocab = self.create_vocab(self.image_to_caption)

        if (
            "word_to_idx.json" in saved_json_files
            and "idx_to_word.json" in saved_json_files
        ):
            self.word_to_idx = json.load(
                open(os.path.join(self.saved_json_path, "word_to_idx.json"))
            )
            self.idx_to_word = json.load(
                open(os.path.join(self.saved_json_path, "idx_to_word.json"))
            )
        else:
            self.word_to_idx, self.idx_to_word = self.create_word_to_idx(self.vocab)

        # if "word_embeddings_400k_glove_200d.json" in saved_json_files:
        #     print("Opening saved word embedding mapping...")
        #     self.word_embd_map = json.load(
        #         open(
        #             os.path.join(
        #                 self.saved_json_path, "word_embeddings_400k_glove_200d.json"
        #             )
        #         )
        #     )
        # else:
        #     print("Creating a word embedding mapping...")
        #     self.create_word_embd_map(
        #         self.glove_path,
        #         os.path.join(
        #             self.saved_json_path, "word_embeddings_400k_glove_200d.json"
        #         ),
        #     )

    def create_word_embd_map(self, glove_path, map_path):
        glove_200d_file = open(glove_path, encoding="utf-8")
        word_embeddings = {}

        for linenum, line in enumerate(glove_200d_file):
            if linenum % 50000 == 0 and linenum > 0:
                print(linenum, "vocab proccessed")
            embd = line.split()
            word_embeddings[embd[0]] = [float(i) for i in embd[1:]]

        word_embd_file = open(map_path, "w")
        json.dump(word_embeddings, word_embd_file)
        word_embd_file.close()

    def create_img_to_captions(self, caption_path):
        captions_json_file = open(caption_path)
        captions_json = json.load(captions_json_file)
        captions_json_file.close()

        imagename_to_captions = {}
        for img in captions_json["images"]:
            captions = []
            for cap in img["sentences"]:
                tokens = [
                    token.lower()
                    for token in cap["tokens"]
                    if len(token) > 1 and not token.isdigit()
                ]
                captions.append("startseq " + " ".join(tokens) + " endseq")
            imagename_to_captions[img["filename"]] = captions

        return imagename_to_captions

    def create_img_to_cap(self, path):
        captions = pd.read_csv(path, sep='|')
        print(captions.columns)

        captions_li = list(zip(captions['image_name'], captions[' comment']))
        img_to_captions = {}
        for cap in captions_li:
            try:
                caption = [i.lower() for i in cap[1].split() if i not in string.punctuation]
                caption = 'startseq ' + ' '.join(caption) + ' endseq'
                if cap[0] in img_to_captions:
                    img_to_captions[cap[0]].append(caption)
                else:
                    img_to_captions[cap[0]] = [caption]
            except:
                print(cap)

        return img_to_captions

    def create_vocab(self, img_to_captions):
        vocab = set()
        for img in img_to_captions:
            captions = img_to_captions[img]
            for cap in captions:
                vocab.update(cap.split())
        return vocab

    def create_word_to_idx(self, vocab):
        word_to_idx = {}
        idx_to_word = {}
        idx = 0
        for word in vocab:
            word_to_idx[word] = idx
            idx_to_word[idx] = word
            idx += 1

        idx_word_file = open(
            os.path.join(self.saved_json_path, "idx_to_word.json"), "w"
        )
        word_idx_file = open(
            os.path.join(self.saved_json_path, "word_to_idx.json"), "w"
        )
        json.dump(idx_to_word, idx_word_file)
        json.dump(word_to_idx, word_idx_file)
        idx_word_file.close()
        word_idx_file.close()

        return word_to_idx, idx_to_word

    def get_maxlen_caption(self, img_to_captions):
        maxLen = 0
        for img in img_to_captions:
            maxLenImg = max([len(cap.split()) for cap in img_to_captions[img]])
            maxLen = max(maxLen, maxLenImg)

        return maxLen


if __name__ == "__main__":
    DATASET_DIR = "./Dataset"
    caption_path = os.path.join(DATASET_DIR, "flickr8k/dataset.json")
    saved_json_path = os.path.join(DATASET_DIR, "Saved_Json")
    glove_path = os.path.join(DATASET_DIR, "GloVe/glove.6B.200d.txt")

    cappre = CaptionPreprocessing(caption_path, saved_json_path, glove_path)
    cappre.preprocess()
    print(cappre.word_to_idx)
    print(cappre.idx_to_word)
