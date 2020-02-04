##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
##############################################################################

import numpy as np
from scipy.stats import entropy

class Dataset(object):

    def __init__(self):
        self.features = None
        self.labels = None

    def read(self, filename):
        with open(filename, 'r') as f:
            n = len(f.readline().split(','))
        self.features = np.genfromtxt(filename, delimiter=',', usecols=list(range(0, n-1)), dtype='float')
        self.labels = np.genfromtxt(filename, delimiter=',', usecols=[n-1], dtype=None)

    def find_maxmin(self):
        ymaxs = np.max(self.features, axis=0)
        ymins = np.min(self.features, axis=0)
        return ymaxs, ymins

    def label_distribution(self):
        return np.unique(self.labels, return_counts=True)

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

    split_col : int
        Keeps track of the column along which we partioned the dataset

    split_val : int
        Keeps track of the value which we used to split the column - the left split
        will contain everything smaller than split_bound and the right split will
        contain everything larger than the split bound.

    left : DecisionTreeClassifier
        The left partion of the dataset will be evaluated along this path.

    right : DecisionTreeClassifier
        The right partition of the dataset will be evaluated along this path.
    """

    def __init__(self, split_col = None, split_val = None, pred_label = None):
        self.is_trained = False
        self.split_col = split_col
        self.split_val = split_val
        self.pred_label = pred_label
        self.left = None
        self.right = None

    def count_item_occ_probs(self, arr):
        probs = np.array([])

        for item in np.unique(arr):
            cnts = np.count_nonzero(arr == item)
            prob = float(cnts) / len(arr)
            probs = np.append(probs, prob)

        return probs

    def optimal_col_split(self, col, y):
        min_ent = float('inf')
        opt_split_val = None
        opt_left_y, opt_right_y = np.array([]), np.array([])

        for split_val in col:
            left_y, right_y = np.array([]), np.array([])

            for val_idx, val in enumerate(col):
                if val < split_val:
                    left_y = np.append(left_y, y[val_idx])
                else:
                    right_y = np.append(right_y, y[val_idx])

            left_probs, right_probs = self.count_item_occ_probs(left_y), self.count_item_occ_probs(right_y)

            left_ent = len(left_y)/len(y) * entropy(left_probs, base = 2)
            right_ent = len(right_y)/len(y) * entropy(right_probs, base = 2)
            split_ent = left_ent + right_ent

            if split_ent < min_ent:
                min_ent = split_ent
                opt_split_val = split_val
                opt_left_y, opt_right_y  = left_y, right_y

        return min_ent, opt_split_val, opt_left_y, opt_right_y

    def optimal_split(self, x, y):

        min_ent = float('inf')
        opt_split_val, opt_split_col = None, None
        opt_left_y, opt_right_y = np.array([]), np.array([])

        for col_idx, col in enumerate(x.T):

            split_ent, split_val, left_y, right_y = self.optimal_col_split(col, y)

            if split_ent < min_ent:
                min_ent = split_ent
                opt_split_val, opt_split_col = split_val, col_idx
                opt_left_y, opt_right_y = left_y, right_y


        left_x, right_x = np.array([]), np.array([])

        for val_idx, val in enumerate(x.T[opt_split_col]):

            if val < opt_split_val:
                if len(left_x) > 0:
                    left_x = np.append(left_x, [x[val_idx]], axis = 0)
                else:
                    left_x = np.array([x[val_idx]])
            else:
                if len(right_x) > 0:
                    right_x = np.append(right_x, [x[val_idx]], axis = 0)
                else:
                    right_x = np.array([x[val_idx]])

        return opt_split_col, opt_split_val, left_x, right_x, opt_left_y, opt_right_y

    def train(self, x, y):

        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        if entropy(self.count_item_occ_probs(y), base = 2) == 0:
            self.pred_label = y[0]
            return

        split_col, split_val, left_x, right_x, left_y, right_y = self.optimal_split(x, y)

        self.split_col = split_col
        self.split_val = split_val

        self.left = DecisionTreeClassifier()
        self.left.train(left_x, left_y)

        self.right = DecisionTreeClassifier()
        self.right.train(right_x, right_y)

        self.is_trained = True

        return self

    def predict_rec(self, x):
        if self.pred_label:
            return self.pred_label
        else:
            if x[self.split_col] < self.split_val and self.left:
                return self.left.predict_rec(x)
            elif self.right:
                return self.right.predict_rec(x)

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

        predictions = []

        for sample in x:
            predictions = np.append(predictions, self.predict_rec(sample))

        return predictions


    def traverse(self):
        if self.pred_label:
            return str(self.pred_label)
        else:
            desc = "Node("
            desc += self.left.traverse() if self.left else "Empty"
            desc += ", "
            desc += self.right.traverse() if self.right else "Empty"
            return desc + ")"

def gof(d1_vals, d1_counts, d2_vals, d2_counts):
    o = scipy.array(d2_counts)
    o_sum = sum(d2_counts)
    e = scipy.array(d1_counts)
    e_sum = sum(d1_counts)

    for i, exp in enumerate(e):
        e[i] = (exp / e_sum) * o_sum

    print(stats.chisquare(o, f_exp=e))

def differ(d1_vals, d1_counts, d2_vals, d2_counts):
    d2_map = {val.decode("utf-8"): count for val, count in zip(d2_vals, d2_counts)}
    res = []

    for val, count in zip(d1_vals, d1_counts):
        val = val.decode("utf-8")
        res.append((val, count, d2_map[val], (count - d2_map[val])*100 / count))

    print(res)


        #######################################################################
        #                 **             MAIN             **
        #######################################################################





'''
d1 = Dataset()
d1.read("data/train_full.txt")
d1_vals, d1_counts = d1.label_distribution()

d2 = Dataset()
d2.read("data/train_noisy.txt")
d2_vals, d2_counts = d2.label_distribution()

gof(d1_vals, d1_counts, d2_vals, d2_counts)
differ(d1_vals, d1_counts, d2_vals, d2_counts)
'''

ds = Dataset()
ds.read("data/toy.txt")

model = DecisionTreeClassifier()
model.train(ds.features, ds.labels)
print(model.traverse())
print(model.predict([[4, 6, 5]]))
