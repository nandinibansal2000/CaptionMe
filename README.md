# CaptionMe

### To run the model
 - Create a folder called Saved_Json inside Dataset directory.
 - Download the pickle files from [this link](https://drive.google.com/drive/folders/1n6RN4HaWz36ei1Jt311e7eAb0-122lDe?usp=sharing)
 - Save an image on the root directory and name it 'Sample.jpg'
 - Download stsb-bert-large model from [this link](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/) and add the path of the model in SimilarityPreprocessor.py
 - Run `python3 main.py <path_to_image>`
 - For example: `python3 main.py sample_images/dog.jpg`

