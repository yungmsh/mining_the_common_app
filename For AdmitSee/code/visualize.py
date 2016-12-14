import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from sklearn.preprocessing import LabelEncoder
from scipy.stats.stats import pearsonr
from summa import textrank
import seaborn as sns

class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""
    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 180);
      this.fig.toolbar.toolbar.attr("y", 540);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

def plotEssays(x, y, labels, titles, cluster_names=None, ms=10, output='notebook'):
    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=x, y=y, label=labels, title=titles))

    #group by cluster
    groups = df.groupby('label')

    #define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
    font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }

    svg.mpld3-figure {
    margin-left: -100px;
    margin-right: -100px}
    """

    #set up colors per clusters using a dict
    # cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: 'b', 6: 'g', 7:'r'}

    #set up cluster names using a dict
    if cluster_names is None:
        cluster_names = {x:x for x in xrange(len(set(labels)))}

    # Plot
    if output == 'notebook':
        fig, ax = plt.subplots(figsize=(14,6)) #set plot size
    elif output == 'app':
        fig, ax = plt.subplots(figsize=(14,8))

    ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=ms,
                         label=cluster_names[name], mec='none')
        ax.set_aspect('auto')
        labels = [i for i in group.title]

        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                           voffset=10, hoffset=10, css=css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())

        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        #set axis as blank
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)

    ax.legend(numpoints=1) #show legend with only one dot
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=7)

    if output=='notebook':
        return mpld3.display()
    elif output=='app':
        html = mpld3.fig_to_html(fig)
        return html

def getTopicLabels(df, type='text'):
    '''
    INPUT: df (pd dataframe), type (str)
    OUTPUT: list

    Gets the highest topic for each essay, outputs as a list of categorical vals
    '''
    le = LabelEncoder()
    text_labels = df.idxmax(axis=1)
    num_labels = le.fit_transform(text_labels)
    if type=='text':
        return text_labels
    if type=='num':
        return num_labels

def getSummaries(essays):
    '''
    INPUT: essays (list)
    OUTPUT: summaries (list)

    Gets essay summaries for each essay.
    '''
    summaries = []
    for i,essay in enumerate(essays):
        try:
            summary = textrank.summarize(essay).replace('\n', ' ')
            summaries.append(summary)
        except:
            summaries.append('Summary not available.')
    return summaries

def showTopicCorr(df, pca, topics, thresh=0.5):
    for i in [0,1]:
        for t in topics:
            a = pca[:,i]
            b = df[t]
            corr = pearsonr(a,b)[0]
            if abs(corr)>thresh:
                print 'PCA {} is {} correlated with {}'.format(i, corr, t)
