from classification import Dataset, LeafNode, DecisionTreeClassifier
from collections import deque
from eval import Evaluator

class DecisionTreePruner(object):

    def __init__(self, tree, valid):
        self.tree = tree
        self.valid = valid
        self.eval = Evaluator()

    def _prune_children(self, parent, node, direction):
        left_pred, right_pred = node.left.prediction, node.right.prediction
        left_count = parent.counts[left_pred]
        right_count = parent.counts[right_pred]

        new_pred = left_pred if left_count > right_count else right_pred

        if direction == 'L':
            parent.left = LeafNode(str(new_pred), prediction=new_pred)
        else:
            parent.right = LeafNode(str(new_pred), prediction=new_pred)

    def _unprune_children(self, parent, node, direction):
        if direction == 'L':
            parent.left = node
        else:
            parent.right = node

    def _get_accuracy(self):
        preds = self.tree.predict(valid.features)
        confusion = self.eval.confusion_matrix(preds, valid.labels)
        accuracy = self.eval.accuracy(confusion)

        return accuracy

    def _find_prunable_nodes(self):

        def prunable(node):
            return type(node.left) is LeafNode and type(node.right) is LeafNode

        queue = deque([(self.tree.root, 1)])
        prunable_nodes = []
        while queue:
            node, depth = queue.popleft()
            if type(node) is not LeafNode:
                for i, child in enumerate([node.left, node.right]):
                    if prunable(child):
                        direction = 'L' if i == 0 else 'R'
                        prunable_nodes.append((node, child, direction, depth))
                    else:
                        queue.append((child, depth + 1))
        return prunable_nodes

    def max_depth(self):
        queue = deque([(self.tree.root, 0)])
        max_depth = 0

        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            if type(node) is not LeafNode:
                queue.append((node.left, depth + 1))
                queue.append((node.right, depth + 1))

        return max_depth


    def prune_tree(self):
        unpruned_accuracy = self._get_accuracy()
        print(unpruned_accuracy)
        improved = True

        while improved:
            max_pruned_accuracy = 0
            prunable_nodes = self._find_prunable_nodes()
            prunable_nodes.sort(key=lambda n: n[3])
            best = None

            for parent, node, direction, depth in prunable_nodes:
                self._prune_children(parent, node, direction)
                pruned_accuracy = self._get_accuracy()

                if pruned_accuracy > max_pruned_accuracy:
                    max_pruned_accuracy = pruned_accuracy
                    best = (parent, node, direction)

                self._unprune_children(parent, node, direction)

            improved = max_pruned_accuracy > unpruned_accuracy

            if improved:
                parent, best_node, direction = best
                print(max_pruned_accuracy)
                unpruned_accuracy = max_pruned_accuracy

                self._prune_children(parent, best_node, direction)

def setup(model, valid, test):
    pruner = DecisionTreePruner(model, valid)
    old = str(pruner.tree)

    old_depth = pruner.max_depth()

    preds = pruner.tree.predict(test.features)
    confusion = eval.confusion_matrix(preds, test.labels)
    old_accuracy = eval.accuracy(confusion)

    pruner.prune_tree()

    preds = pruner.tree.predict(test.features)
    confusion = eval.confusion_matrix(preds, test.labels)
    new_accuracy = eval.accuracy(confusion)

    new_depth = pruner.max_depth()

    return old_accuracy, new_accuracy, old_depth, new_depth


if __name__ == "__main__":
    headers = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
               "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

    valid = Dataset("data/validation.txt")
    test = Dataset("data/test.txt")
    eval = Evaluator()

    full_model = DecisionTreeClassifier(headers)
    full_model.deserialise_model("data/model_full.pickle")

    noisy_model = DecisionTreeClassifier(headers)
    noisy_model.deserialise_model("data/model_noisy.pickle")

    print(setup(full_model, valid, test))
    print()
    print(setup(noisy_model, valid, test))
