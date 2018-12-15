import numpy as np

def getAP(conf,labels):
    assert len(conf)==len(labels)
    sortind = np.argsort(-conf)
    tp = labels[sortind]==1; fp = labels[sortind]!=1
    npos = np.sum(labels);

    fp = np.cumsum(fp).astype('float32'); tp = np.cumsum(tp).astype('float32')
    rec=tp/npos; prec=tp/(fp+tp)
    tmp = (labels[sortind]==1).astype('float32')

    return np.sum(tmp*prec)/npos

def getClassificationMAP(confidence,labels):
    ''' confidence and labels are of dimension n_samples x n_label '''

    AP = []
    for i in range(np.shape(labels)[1]):
       AP.append(getAP(confidence[:,i], labels[:,i]))
    return 100*sum(AP)/len(AP)
