
# Import packages
import numpy as np
import pandas as pd

# Analytics/Metrics
from sklearn.feature_extraction.text import CountVectorizer


def write_target_descriptive_categories(value):
    """ Replaces values in the CATEGORY column with numeric values. """
    if value == 1:
        return "Entertainment"
    elif value == 2:
        return "Business"
    elif value == 3:
        return "Science.Technology"
    elif value == 4:
        return "Health"
    else:
        return "ERROR"
        # return 999




def create_word_frequency_table(df, category):

    # Subset to category
    df = df[df['Descriptive Category']==category]
    #Create DTM
    cv = CountVectorizer(ngram_range = (1,3), stop_words='english')
    dtm = cv.fit_transform(df['clean_title'])
    words = np.array(cv.get_feature_names_out())

    #Look at top 10 most frequent words
    freqs=dtm.sum(axis=0).A.flatten()
    index=np.argsort(freqs)[-20:]

    # Construct dataframe
    WordFreq = pd.DataFrame.from_records(list(zip(words[index], freqs[index])))
    WordFreq.columns = ['Word', 'Frequency']
    return WordFreq