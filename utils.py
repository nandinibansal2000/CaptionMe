from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def get_common_images(img_to_captions, img_to_image_features):
    img_to_image = {}
    img_to_cap = {}
    for img in img_to_image_features:
        if img in img_to_captions:
            img_to_image[img] = img_to_image_features[img]

    for img in img_to_captions:
        if img in img_to_image:
            img_to_cap[img] = img_to_captions[img]

    return img_to_image, img_to_cap


def train_test_split_(img_to_captions, img_to_image_features, test_size=0.3):
    img_to_image_features, img_to_captions = get_common_images(img_to_captions, img_to_image_features)
    captions = sorted([item for item in list(img_to_captions.items())])

    image_features = sorted([item for item in list(img_to_image_features.items())])
    train_captions, test_captions = train_test_split(
        captions, test_size=test_size, random_state=42
    )
    train_images, test_images = train_test_split(
        image_features, test_size=test_size, random_state=42
    )

    assert list(zip(*train_captions))[0] == list(zip(*train_images))[0]
    assert list(zip(*test_captions))[0] == list(zip(*test_images))[0]
    return train_images, test_images, train_captions, test_captions


def build_embedding_matrix(vocab_size, embedding_dim, word_to_idx, embdgs_map):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_to_idx.items():
        embedding_vector = embdgs_map.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix



def greedySearch(photo, maxlen, word_to_idx, idx_to_word, model):
    in_text = 'startseq' 
    # print(idx_to_word.keys())   
    photo = np.array([photo])
    for i in range(maxlen):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = np.array(pad_sequences([sequence], maxlen=maxlen))
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
