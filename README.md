# CaptionMe

### Image caption generation
 - We propose to build an Image captioning system for social media platforms in this project. The system will try to find a suitable quote which aptly describes the image to be posted. Using our system, we aim to lessen the burden on the user’s mind and let our system’s creativity take over.
 -  Image  caption  generation  is  a  common  problem. Most of the current work focuses on generating textual description for a given image.  However, this kind of approach won’t be very useful when it comes to social media since the caption that is descriptive of the given image isn’t normally social media relevant. 
 
### To run the model
 - Create a folder called Saved_Json inside Dataset directory.
 - Download the pickle files from [this link](https://drive.google.com/drive/folders/1n6RN4HaWz36ei1Jt311e7eAb0-122lDe?usp=sharing)
 - Save an image on the root directory and name it 'Sample.jpg'
 - Download stsb-bert-large model from [this link](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/) and add the path of the model in SimilarityPreprocessor.py
 - Run `python3 main.py <path_to_image>`
 - For example: `python3 main.py sample_images/dog.jpg`

