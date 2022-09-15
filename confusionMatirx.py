import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix        # confusion matrix 사용을 위한 라이브러리
import itertools
# get data
def getConfusionData(data):
    return confusion_matrix(data[0],data[1])
# get Matrix
def draw_confusionMatrix(_label,confusionData,_normalize=False,_title='Confusion Matrix',_cmap=plt.cm.get_cmap('Purples')):
    # draw confusion matrix
    conM = confusion_matrix(confusionData[0], confusionData[1])
    plt.imshow(conM, interpolation='nearest', cmap=_cmap)
    plt.title(_title)
    plt.colorbar()
    marks = np.arange(len(_label))
    nlabels = []
    for k in range(len(conM)):
        n = sum(conM[k])
        nlabel = '{0}(n={1})'.format(_label[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, _label)
    plt.yticks(marks, nlabels)
    
    thresh = conM.max() / 2.
    if _normalize:
        for i, j in itertools.product(range(conM.shape[0]), range(conM.shape[1])):
            plt.text(j, i, '{0}%'.format(conM[i, j] * 100 / n), horizontalalignment="center", color="white" if conM[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(conM.shape[0]), range(conM.shape[1])):
            plt.text(j, i, conM[i, j], horizontalalignment="center", color="white" if conM[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
