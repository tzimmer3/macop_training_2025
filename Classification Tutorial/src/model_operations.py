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


# ================== #
#  Tokenizer
# ================== #

def count_tokens(string):
    """
    Count the number of words in a string.
    """
    return len(string.split(" "))


# ================== #
#  Target Column
# ================== #

def clean_target(value):
    """ Replaces values in the CATEGORY column with numeric values. """
    if value == 'e':
        return 1
    elif value == 'b':
        return 2
    elif value == 't':
        return 3
    elif value == 'm':
        return 4
    else:
        return "ERROR"
        # return 999


def write_target_descriptive_categories(value):
    """ Replaces values in the CATEGORY column with numeric values. """
    if value == 1:
        return "Entertainment"
    elif value == 2:
        return "Business"
    elif value == 3:
        return "Science/Technology"
    elif value == 4:
        return "Health"
    else:
        return "ERROR"
        # return 999