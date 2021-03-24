from sentence_transformers import SentenceTransformer, util
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.corpus import words

class SimilarityPreprocessor:

    '''
    This class is used for finding similarity between text info and quotes.
    '''



    def __init__(self):
        self.model = SentenceTransformer('PATH_TO_stsb-bert-large')



    def getQuoteMapping(self,df):
        """
        Returns the Quote to Quote ID Mapping and Quote ID to Quote Mapping.
          Parameters:
                df(pandas.DataFrame) : DataFrame of Quotes, Author and Tags
          Returns:
                quote_qid(dict) : Quote to Quote ID Mapping
                qid_quote(dict) : Quote ID to Quote Mapping
        """
        qid_quote = dict()
        quote_qid = dict()
        for i in range(len(df)):
            qid_quote[i+1] = df["Quote"].iloc[i]
            quote_qid[df["Quote"].iloc[i]] = i+1
        return quote_qid,qid_quote



    def getEmbeddings(self,df):
        """
        Returns the Quote ID to Vector Embeddings Mapping.
          Parameters:
                df(pandas.DataFrame) : DataFrame of Quotes, Author and Tags
          Returns:
                qid_vector(dict) : Quote ID to Vector Embeddings Mapping
        """
        qid_vector = dict()
        for i in range(len(df)):
            quote_embedding = self.model.encode(df["Quote"].iloc[i])
            qid_vector[i+1] = quote_embedding
        return qid_vector



    def getTagMapping(self,df):
        """
        Returns the Tags to Quote ID Mapping.
          Parameters:
                df(pandas.DataFrame) : DataFrame of Quotes, Author and Tags
                quote_qid(dict) : Quote to Quote ID Mapping
          Returns:
                tags_qid(dict) : Tags to Quote ID Mapping
        """
        tags_qid = dict()

        en_words = dict()
        for i in words.words():
          en_words[i] = 1

        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()

        for i in range(len(df)):
            text = df['Tags'].iloc[i]
            qid_ = i+1
            text = text.split(",")
            for i_ in text:
                i = i_.strip()
                i = i.lower()
                i = lemmatizer.lemmatize(i)
                if("-" in i or len(i.split(" "))>1 or len(i)<=2 or (en_words.get(i,0)==0)):
                    continue
                else:
                    i = ps.stem(i)
                    tags_qid[i] = tags_quotes.get(i,[])
                    tags_qid[i].append(qid_)
        return tags_qid



    def getSimilarityScore(self,query_embedding, quote_embedding):
        """
        Returns the Cosine Similarity Score between Query Embedding and Quote Embedding.
          Parameters:
                query_embedding(tensorflow.float32) : Tensorflow Vector of Image Text
                quote_embedding(tensorflow.float32) : Tensorflow Vector of Quote
          Returns:
                score(float) : Cosine Similarity Score between Query Embedding and Quote Embedding
        """
        score = util.pytorch_cos_sim(query_embedding,quote_embedding) 
        return score



    def getSimilarQuotes(self,imgtext,qids,k,qid_vector):
        """
        Returns the Top k Quotes based on Cosine Similarity Score.
          Parameters:
                imgtext(str) : Text description of image
                qids(list) : Qoute IDs to be processed
                k(int) : The top number of quotes to be returned
                qid_vector(dict) : Quote ID to Vector Embeddings Mapping
          Returns:
                TopKScores(list) : Top k tuples of Score along with Quote ID
        """
        scores = []
        query_embedding = self.model.encode(imgtext)
        for i in range(len(qids)):
            qid = qids[i]
            quote_embedding = qid_vector[qid]
            scores.append((self.getSimilarityScore(query_embedding,quote_embedding),qid))
        scores = sorted(scores, key=lambda x:(-x[0]))
        TopKScores = scores[:k]
        return TopKScores



    def getTopKQuotes(self,imgtext,tags_qid,k,qid_vector):
        """
        Returns the Top k Quotes based on Cosine Similarity Score.
          Parameters:
                imgtext(str) : Text description of image
                tags_qid(dict) : Tags to Quote ID Mapping
                k(int) : The top number of quotes to be returned
                qid_vector(dict) : Quote ID to Vector Embeddings Mapping
          Returns:
                score_qid(list) : Top k tuples of Score along with Quote ID
        """
        i_ = imgtext.split(' ')
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
        qids = []
        for j in i_:
            j_ = j.strip()
            j_ = j_.lower()
            j_ = lemmatizer.lemmatize(j_)
            j_ = ps.stem(j_)
            if (j_ in tags_qid):
                lst_ =  tags_qid[j_]
                qids.extend(lst_)

        if(len(qids)!=0):
            qids = list(set(qids))
            score_qid = self.getSimilarQuotes(imgtext,qids,k,qid_vector)
            return score_qid
        else:
            qids = [i+1 for i in range(len(qid_vector))]
            score_qid = self.getSimilarQuotes(imgtext,qids,k,qid_vector)
            return score_qid
    









