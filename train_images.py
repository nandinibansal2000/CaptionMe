from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from utils import train_test_split_, get_common_images, build_embedding_matrix

def generate_data(train_captions, train_images, word_to_idx, maxlen, batchSize=5):
    X_image, X_capSeq, y = [], [], []
    train_images = dict(train_images)
    n = 0
    while True:
        for img, desc_list in train_captions:        
            photo = train_images[img]
            photo = photo.reshape(photo.shape[0], )
            for desc in desc_list:
                seq = [word_to_idx[word] for word in desc.split(' ') if word in word_to_idx]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=maxlen)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X_image.append(photo)
                    X_capSeq.append(in_seq)
                    y.append(out_seq)

            n += 1
            if n == batchSize:
                yield [np.array(X_image), np.array(X_capSeq)], np.array(y)
                X_image, X_capSeq, y = [], [], []
                n = 0


def get_model(vocab_size, embedding_dim, embedding_matrix):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(maxlen,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    print(model.summary())
    print(model.layers[2])

    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train():
    embedding_dim = 200
    epochs = 30
    batchSize = 32
    steps = len(train_captions)//batchSize

    image_features_path = './Dataset/image_features.pkl'
    caption_path = './Dataset/flickr8k/dataset.json'
    saved_json_path = './Dataset/Saved_Json'
    glove_path = './Dataset/GloVe/glove.6B.200d.txt'
    img_to_image_features = pickle.load(open(image_features_path, 'rb'))

    captionPreprocess = CaptionPreprocessing(caption_path, saved_json_path, glove_path)
    captionPreprocess.preprocess()
    img_to_caption = captionPreprocess.image_to_caption
    vocab_size = len(captionPreprocess.vocab)
    word_to_idx = captionPreprocess.word_to_idx
    idx_to_word = captionPreprocess.idx_to_word
    embdgs_map = captionPreprocess.word_embd_map
    maxLen = captionPreprocess.maxLen

    train_images, test_images, train_captions, test_captions = train_test_split_(img_to_caption, img_to_image_features, test_size=0.3)

    embedding_matrix = build_embedding_matrix(vocab_size, embedding_dim, word_to_idx, embdgs_map)


    model = get_model(vocab_size, embedding_dim, embedding_matrix)

    for i in range(epochs):
        generator = generate_data(train_captions, 
                                train_images,
                                word_to_idx, 
                                maxLen, 
                                batchSize=batchSize)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        p = os.path.join('./Model', 'model_' + str(i) + '.h5')
        model.save(p)

    p = os.path.join('./Model', 'model_final.h5')
    model.save(p)

