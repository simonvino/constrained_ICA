import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import sawtooth, square


def generate_artificial_signals(n_sig=5, n_samples=10000, save_signals=False):
    """Generate aritificial signals, references and random signal mixtures.

    Parameters
    ----------
    n_sig : int, default=5
        Number of signals.

    n_samples : int, default=10000
        Number of samples per signal.

    save_signal : boolean, default=False
        Save generated signals or not.

    Returns
    -------
    Sigs : ndarray of shape (n_sig, n_samples)
        Generated source signals.

    Refs : ndarray of shape (n_sig, n_samples)
        References for source signals.
        
    X : ndarray of shape (n_sig, n_samples)
        Random mixture of signals.
    """
    # Generate source signals and references.
    Sigs, Refs = [], []
    for n in range(1, n_sig+1):
        freq = n/100
        inp = np.arange(0, n_samples)
        if (n % 3) == 0:
            Sigs.append(sawtooth(inp*np.sqrt(n*4)/10))
            Refs.append(1.5*sawtooth(inp*np.sqrt(n*4)/10))
        else:
            Sigs.append(np.sin(2*np.pi*freq*inp))
            Refs.append(square(2*np.pi*freq*inp))
            
    Sigs, Refs = np.stack(Sigs, axis=0),  np.stack(Refs, axis=0)

    # Create mixtues.
    np.random.seed(42)
    A = np.random.rand(n_sig, n_sig)
    X = np.matmul(A, Sigs)
    
    # Save data.
    save_signals = False
    data_dir = './data/'
    if save_signals:
        print('Shape signals: ', Sigs.shape)
        np.savetxt(data_dir + 'signals.txt', Sigs, delimiter=',')
        np.savetxt(data_dir + 'references.txt', Refs, delimiter=',')
        np.savetxt(data_dir + 'mixtures.txt', X, delimiter=',')
        
    return Sigs, Refs, X


def plot_history(history, xlim=None, thres=None,
                 plt_metrics=['mu', 'similarity', 'kurtosis', 'negentropy']):
    """Function for plotting the training history.

    Parameters
    ----------
    history : list of length (n_iterations)
        List tracking metrics during the ICA training.

    xlim : int, default=None
        Limit of iterations to plot.

    thres : int, default=None
        Similarity threshold.
    
    plt_metrics : list 
        List of all metrics to plot.
    """
    
    history_df = pd.DataFrame(history)
    history_df 
    
    n_ICs = len(history_df['IC'].unique())
    
    fig, axs = plt.subplots(len(plt_metrics), figsize=(10, 2.5*len(plt_metrics)))
    for n_val, val in enumerate(plt_metrics):
        sns.lineplot(data=history_df, y=val, x='iteration', errorbar='sd',
                     ax=axs[n_val], hue='IC', palette="rainbow", 
                     legend='full', alpha=0.5)
        
        # Plot horizontal line for threshold.
        if val == 'similarity' and thres is not None:
            axs[n_val].axhline(y=thres, color='grey', label='thr',
                               linestyle='dashed', alpha=0.5)
            
        axs[n_val].get_xaxis().get_label().set_visible(False)
        axs[n_val].legend(loc='upper right', ncol=(n_ICs//10)+1)
        axs[n_val].set_xlim([0, xlim])
        sns.despine(offset=10)

    axs[n_val].get_xaxis().get_label().set_visible(True)
    fig.suptitle('Training history')
    plt.tight_layout(h_pad=2)
    
    
def plot_components(Sigs, Refs, C, labels=['Sig', 'Ref', 'C'], 
                    titles=[None, None], n_points=None):
    """Functions for plotting components.

    Parameters
    ----------
    Sigs : array-like of shape (n_features, n_samples)
        Signal or source components.

    Refs : array-like of shape (n_refs, n_samples)
        Reference signals, where `n_samples` is the number of samples
        and `n_refs` is the number of features.
        
    C : array-like of shape (n_comp, n_samples)
        Components to visualize.
    """
    n_sig = Sigs.shape[0]
    palette = sns.color_palette("rainbow", n_sig+1)
    
    fig, axs = plt.subplots(Sigs.shape[0], 2, figsize=(18, n_sig*2.5))
    for n in range(n_sig):

        # Plot signal.
        sns.lineplot(data=Sigs[n, :n_points], ax=axs[n, 0], marker="o",
                     markersize=4, color=palette[n], label='{}_{}'.format(labels[0], n),
                     linewidth=2)
        
        # Plot reference.
        try:  # Check if reference is available.
            sns.lineplot(data=Refs[n, :n_points], ax=axs[n, 0],
                         markersize=4, color=palette[n+1], label='{}_{}'.format(labels[1], n), 
                         alpha=0.5)
        except:
            pass
        
        axs[n, 0].get_xaxis().set_visible(False)
        axs[n, 0].set_ylabel('Amplitude')
        axs[n, 0].legend(loc='upper right')

        # Plot component.
        try:  # Check if component is available. 
            sns.lineplot(data=C[n, :n_points], ax=axs[n, 1], marker="o", 
                         markersize=4, color='grey', label='{}_{}'.format(labels[2], n))
            axs[n, 1].get_xaxis().set_visible(False)
            axs[n, 1].legend(loc='upper right')
        except:
            pass

    # Set titles.
    axs[0, 0].set_title(titles[0])
    axs[0, 1].set_title(titles[1])
    
    # Add xlabels.
    sns.despine()
    axs[n, 0].get_xaxis().set_visible(True)
    axs[n, 0].set_xlabel('Sample')
    axs[n, 1].get_xaxis().set_visible(True)
    axs[n, 1].set_xlabel('Sample')
    
    
def plot_pca_variance(variances, xlim=None, ylim=None):
    """Functions for plotting explained variance of principal components.

    Parameters
    ----------
    variances : array-like of shape (n_comp)
        Signal or source components.

    xlim : int
        Number of PCs to plot.
        
    ylim : int
        Maximum variance to plot.        
    """  
    
    variances_df = pd.DataFrame(variances, columns=['explained variance'])
    variances_df['component'] = variances_df.index
    
    ax = sns.lineplot(variances_df, x='component',
                      y='explained variance', marker="o")
    plt.ylim([0, ylim])
    plt.xlim([0, xlim])
    sns.despine(offset=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    
def negentropy(sig): 
    """Compute approximated negentropy of signal.

    Parameters
    ----------
    sig : array-like of shape (n_samples)
        Signal.        
    """ 
    sig_std = sig.std(axis=1)
    p_gauss = np.random.normal(loc=np.zeros(sig.shape), scale=np.tile(sig_std, (sig.shape[1], 1)).transpose())
    rou = np.mean(np.log(np.cosh(sig)) - np.log(np.cosh(p_gauss)), axis=1)
    return rou
    
    