# BERT
from sentence_transformers import SentenceTransformer
from joblib import load


# ================== #
#  Get Embeddings
# ================== #

def get_embeddings(text=None, model=None):
    """
    Embed a string using a sentence BERT model.
    """
    if model==None:
        model = load('./models/SentBERTmodel.pkl')

    return model.encode(text)