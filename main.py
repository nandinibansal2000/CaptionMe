from imagePreprocessing import *
from evaluate import *
import pickle
import keras
import numpy as np
from SimilarityPreprocessor import SimilarityPreprocessor
import sys

# Extracting textual information from the image
preprocessing = PreProcessing()

# Encoded image
image_name = sys.argv[1]
print(image_name)
encode_img = preprocessing.getEncode(image_name)
test_images = [('image', encode_img)]

# Directory Path
saved_json_path = './Dataset/Saved_Json'

# Loading helper dictionaries
caption_summary =  pickle.load(open(os.path.join(saved_json_path, 'caption_summary.pkl'),'rb'))
word_to_idx = caption_summary['word_to_idx']
idx_to_word = caption_summary['idx_to_word']
maxLen = caption_summary['maxLen']

model_path = os.path.join(saved_json_path, 'model_final.h5')
model = keras.models.load_model(model_path)

predicted_text = get_pred(test_images, maxLen, word_to_idx, idx_to_word, model)[0]
print(predicted_text)
# Extraction top k quotes
similarity_preprocessor = SimilarityPreprocessor()

quote_summary =  pickle.load(open(os.path.join(saved_json_path, 'quote_summary.pkl'),'rb'))
tags_qid = quote_summary['tag_qids']
qid_vector = quote_summary['qid_vector']
qid_quote = quote_summary['qid_quote']

# Number of quotes to be extracted
k = 3

top_k_qid= similarity_preprocessor.getTopKQuotes(predicted_text, tags_qid,k,qid_vector)

for i in range(len(top_k_qid)):
	i_ = top_k_qid[i][1]
	print("Quote " + str(i+1) +":", qid_quote[i_])
