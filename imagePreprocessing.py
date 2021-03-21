import warnings

warnings.filterwarnings(action="ignore")
import numpy as np
from numpy import array
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input


class PreProcessing:
    """
    This is a class for doing preprocessing on image.
    """

    def __init__(self):
        self.model = InceptionV3(weights="imagenet")
        self.model_ = Model(
            inputs=self.model.input, outputs=self.model.layers[-2].output
        )

    def preprocess(self, img):
        """
        Returns the preprocessed image

          Parameters:
                  img(str): input image path
          Returns:
                  img_(): processed image
        """
        img_ = image.load_img(img, target_size=(299, 299))
        img_ = image.img_to_array(img_)
        img_ = np.expand_dims(img_, axis=0)
        img_ = preprocess_input(img_)
        return img_

    def getEncode(self, img):
        """
        Returns the feature vector of input image

          Parameters:
                  img(str): input image path
          Returns:
                  fv(): feature vector
        """
        img_ = self.preprocess(img)
        fv = self.model_.predict(img_)
        fv = fv.reshape(-1, 1)
        return fv
