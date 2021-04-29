import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def plot_signal_background(df, input_feature, fig=None, ax=None, nbins=20):
    if not (fig and ax):
        fig, ax = plt.subplots()
    sig = df.loc[df['is_signal_new'] == 1][input_feature].values
    bkg = df.loc[df['is_signal_new'] == 0][input_feature].values
    min_val = min(np.min(sig), np.min(bkg))
    max_val = max(np.max(sig), np.max(bkg))
    _, bin_edges, _ = ax.hist(sig, bins=20, range=(min_val, max_val), color='b', histtype='step',
                              linewidth=2, label='top jets')
    _, bin_edges, _ = ax.hist(bkg, bins=bin_edges, color='r', histtype='step', linewidth=2,
                              label='gluon jets')
    ax.set_xlim(min_val, max_val)
    ax.set_xlabel(input_feature)
    ax.set_ylabel('number of events')
    ax.grid()
    ax.legend()
    return fig, ax


def plot_accuracy(model_history):
    """
    Plot training and validation accuracy.

    Parameters
    ----------
    ax:       A blank ``matplotlib.axis.Axis`` object that the plot should be drawn to.
    training: A ``Training`` object that's ``evaluate`` method has already been executed.

    Returns
    -------
    The processed ``matplotlib.axis.Axis`` object.
    """
    fig, ax = plt.subplots()
    ax.plot(model_history.epoch, model_history.history['accuracy'], color='r', linewidth=2, label='accuracy training')
    ax.plot(model_history.epoch, model_history.history['val_accuracy'], color='b', linewidth=2, label='accuracy validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.grid()
    ax.legend()
    return fig, ax


def plot_loss(model_history):
    """
    Plot training and validation loss.

    Parameters
    ----------
    ax:       A blank ``matplotlib.axis.Axis`` object that the plot should be drawn to.
    training: A ``Training`` object that's ``evaluate`` method has already been executed.

    Returns
    -------
    The processed ``matplotlib.axis.Axis`` object.
    """
    fig, ax = plt.subplots()
    ax.plot(model_history.epoch, model_history.history['loss'], color='r', linewidth=2, label='loss training')
    ax.plot(model_history.epoch, model_history.history['val_loss'], color='b', linewidth=2, label='loss validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    return fig, ax


def plot_network_output(data_test, labels_test, labels_predicted, nbins=20):
    """
    Plot histogram of the network output values and distinguish between true signal and true background events.

    Parameters
    ----------
    ax:               A blank ``matplotlib.axis.Axis`` object that the plot should be drawn to.
    training:         A ``Training`` object that's ``evaluate`` method has already been executed.
    nbins (optional): Number of bins that the network output values are filled into. Default is ``20``.

    Returns
    -------
    The processed ``matplotlib.axis.Axis`` object.
    """
    fig, ax = plt.subplots()
    labels_predicted_sig = labels_predicted[np.where(labels_test == 1)]
    labels_predicted_bkg = labels_predicted[np.where(labels_test == 0)]
    _, bin_edges, _ = ax.hist(labels_predicted_sig, bins=nbins, range=(0., 1.), color='b', histtype='step',
                              linewidth=2, label='top jets')
    _, _, _ = ax.hist(labels_predicted_bkg, bins=bin_edges, color='r', histtype='step',
                      linewidth=2, label='gluon jets')
    ax.set_xlim(0., 1.)
    ax.set_xlabel('network output value')
    ax.set_ylabel('number of events')
    ax.grid()
    ax.legend()
    return fig, ax


def plot_roc_curve(labels_test, labels_predicted, nbins=20):
    """
    Plot histogram of the network output values and distinguish between true signal and true background events.

    Parameters
    ----------
    ax:               A blank ``matplotlib.axis.Axis`` object that the plot should be drawn to.
    training:         A ``Training`` object that's ``evaluate`` method has already been executed.
    nbins (optional): Number of bins that the network output values are filled into. Default is ``20``.

    Returns
    -------
    The processed ``matplotlib.axis.Axis`` object.
    """
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(labels_test, labels_predicted)
    roc_points = np.array([np.array([tpr[i], 1. - fpr[i]]) for i in range(len(tpr))])
    roc_auc = roc_auc_score(labels_test, labels_predicted)
    ax.plot(roc_points[:, 0], roc_points[:, 1], color='g', linewidth=2,
            label='ROC curve, ROC-AUC {0:.5f}'.format(roc_auc))
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_xlabel('signal efficiency')
    ax.set_ylabel('background rejection')
    ax.grid()
    ax.legend()
    return fig, ax


def plot_overlayed_jet_images(df, category_class):
    """
    Plot jet images from randomly selected top and gluon jet objects.

    Parameters
    ----------
    ax:             A blank ``matplotlib.axis.Axis`` object that the plot should be drawn to.
    data_container: A ``DataContainerDNN`` object that contains a training data frame of the constituent's dataset.
    category_class: Can be set to ``top`` or ``gluon`` for showing top or gluon jet images.
    """
    fig, ax = plt.subplots()
    pixels = ['img_{0:d}'.format(i) for i in range(1600)]
    if category_class == 'top':
        image_rows = df.loc[df['is_signal_new'].values == 1].sample(n=10000)[pixels].values
        colormap_name = 'Blues'
    elif category_class == 'gluon':
        image_rows = df.loc[df['is_signal_new'].values == 0].sample(n=10000)[pixels].values
        colormap_name = 'Reds'
    else:
        raise ValueError('Valid options for \'category_class\' are either \'top\' or \'gluon\'.')
    overlay_image = np.reshape(np.sum(image_rows, axis=0), (40, 40))
    del image_rows
    ax.imshow(overlay_image, cmap=colormap_name)
    ax.set_xlabel('$\\eta$ bin index')
    ax.set_ylabel('$\\phi$ bin index')
    ax.set_title('Overlayed {0:s} Jet Images'.format(category_class))
    return fig, ax


