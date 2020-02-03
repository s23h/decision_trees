import numpy as np
import scipy
from scipy import stats

class Dataset(object):

    def __init__(self):
        self.features = None
        self.labels = None

    def read(self, filename):
        with open(filename, 'r') as f:
            n = len(f.readline().split(','))
        self.features = np.loadtxt(filename, delimiter=',', usecols=(range(0, n-1)), dtype='float')
        self.labels = np.loadtxt(filename, delimiter=',', usecols=(n-1), dtype='S12')

    def find_maxmin(self):
        ymaxs = np.max(self.features, axis=0)
        ymins = np.min(self.features, axis=0)
        return ymaxs, ymins

    def label_distribution(self):
        return np.unique(self.labels, return_counts=True)


class DecisionNode(object):

    def __init__(self, split_col=None, split_val=None):
        self.split_col = split_col
        self.split_val = split_val
        self.left = None
        self.right = None
        self.prediction = None

    def __str__(self):
     return ""


class LeafNode(DecisionNode):

    def __str__(self):
        return ""


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self):
        self.is_trained = False
        self.root = None

    def calc_entropy(self, y):
        num_lbls = len(y)

        if num_lbls <= 1:
            return 0

        val, cnts = np.unique(y, return_counts = True)
        probs = cnts.astype(float) / num_lbls
        num_cl = np.count_nonzero(probs)

        if  num_cl <= 1:
            return 0

        entropy = 0

        for prob in probs:
            entropy -= prob * math.log(prob, 2)

        return entropy

    def find_opt_split(self, x, y):
        min_ent = float('inf')
        opt_val, opt_col_idx = None, None
        opt_left, opt_right = None, None

        classes = np.unique(y)
        if len(classes) == 1:
            return None

        for i, col in enumerate(x.T):
            for val in col:
                left, right = self.split(i, val, y)
                split_ent = len(left_y)/len(y) * self.calc_entropy(left_y) + len(right_y)/len(y) * self.calc_entropy(right_y)

                if split_ent < min_ent:
                    min_ent = split_ent
                    opt_val, opt_col_idx = val, i
                    opt_left, opt_right = left, right

        return opt_left, opt_right, opt_val, opt_col_idx



    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
        An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        left_y, right_y, opt_val, opt_col_idx = self.find_opt_split(x, y)
        self.root = DecisionNode(opt_col_idx, opt_val)

        self.root.left, self.root.right = self.split(x, y)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True



        return self


    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)


        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################


        # remember to change this if you rename the variable
        return predictions
