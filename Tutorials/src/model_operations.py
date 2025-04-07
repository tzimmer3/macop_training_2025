# BERT
import matplotlib.pyplot as plt
from joblib import load


# ================== #
#  Get Embeddings
# ================== #

def get_embeddings(text=None, model=None):
    """
    Embed a string using a sentence BERT model.
    """
    if model==None:
        model = load('../Classification Tutorial/models/SentBERTmodel.pkl')

    return model.encode(text)


# ================== #
#  Tokenizer
# ================== #

def create_headmap(matrix, title):
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set the title of the heatmap
    ax.set_title(title, fontsize=20)

    # Set up the heatmap
    heatmap = plt.pcolor(matrix, 
                        linewidth=2.5, 
                        edgecolor="white", 
                        cmap='Purples', 
                        alpha=0.75)


    # Annotate the heatmap with scores
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % matrix[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    )

    plt.yticks(ticks=[], labels=[])

    
    return fig, ax


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