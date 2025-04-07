import pandas as pd
import matplotlib.pyplot as plt


# Functions

# ====== #
# Data Processing
# ====== #

def filter_on_coordinates(data, x_low=None, x_high=None, y_low=None, y_high=None):
    """ Takes a dataframe and filters based on criteria for two columns"""
    filtered_data = data[(data['x']>=x_low)&(data['x']<=x_high)]
    filtered_data = filtered_data[(filtered_data['y']>=y_low)&(filtered_data['y']<=y_high)]
    return filtered_data

def filter_on_category(data, criteria):
    """ Takes a dataframe and filters based on criteria for a column"""
    return data[data['Target']==criteria]


def filter_on_two_categories(data, criteria):
    """ Takes a dataframe and filters based on criteria for a column"""
    return data[(data['Descriptive Category Label']==criteria[0])|(data['Descriptive Category Label']==criteria[1])]


# ====== #
# Visuals
# ====== #

def create_scatter_plot(visualization_dataset):
    """
    Creates a scatter plot from a pandas dataframe.  Dataframe must have columns: ['x','y','Hits Labeled']
    """

    # Get color for each data point
    colors = {'Correct Predictions': 'gold', 'Incorrect Predictions': 'rebeccapurple'}
    color_list = [colors[group] for group in visualization_dataset['Hits Labeled']]

    # Create a scatter plot with color-coding based on 'categorical_variable'
    ax = visualization_dataset.plot.scatter('x',
                    'y',
                    c=color_list,
                    grid=True,
                    figsize=(12,10))

    # Create legend handles, labels for each group and add legend to the plot
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=colors['Correct Predictions'], label='Correct Predictions'),
        mpatches.Patch(color=colors['Incorrect Predictions'], label='Incorrect Predictions'), # add as many as needed
    ]
    ax.legend(handles=legend_handles, loc='upper left')

    # Add title and labels ('\n' allow us to jump rows)
    ax.set_title("News Dataset Predictions and Similarities", name='Sans', fontsize = 28, color='indigo')
    ax.set_xlabel('Visually check for patterns in the data.\n  Points close together are similar news stories.\n  Points are colored by correct/incorrect model predictions.')
    ax.set_ylabel('')

    # Hide x and y ticks (because they dont mean anything)
    plt.xticks(color='w') 
    plt.yticks(color='w')

    plt.show()


def create_scatter_plot_categories(visualization_dataset, criteria):
    """
    Creates a scatter plot from a pandas dataframe.  Dataframe must have columns: ['x','y','Hits Labeled']
    """

    # Get color for each data point
    colors = {criteria[0]: 'blue', criteria[1]: 'green'}
    color_list = [colors[group] for group in visualization_dataset['Descriptive Category Label']]

    # Create a scatter plot with color-coding based on 'categorical_variable'
    ax = visualization_dataset.plot.scatter('x',
                    'y',
                    c=color_list,
                    grid=True,
                    figsize=(12,10))

    # Create legend handles, labels for each group and add legend to the plot
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=colors[criteria[0]], label=criteria[0]),
        mpatches.Patch(color=colors[criteria[1]], label=criteria[1]), # add as many as needed
    ]
    ax.legend(handles=legend_handles, loc='upper left')

    # Add title and labels ('\n' allow us to jump rows)
    ax.set_title(f"{criteria[0]} vs {criteria[1]} Title Embeddings", name='Sans', fontsize = 28, color='black')
    ax.set_xlabel('Visually check for patterns in the data.\n  Points close together are similar news titles.\n  Points are colored by model prediction categories.')
    ax.set_ylabel('')

    # Hide x and y ticks (because they dont mean anything)
    plt.xticks(color='w') 
    plt.yticks(color='w')

    plt.show()