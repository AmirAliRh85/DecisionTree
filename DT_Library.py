class Node:
    """
        Represents a node of the decision tree.
    
        Internal node: feature_index, threshold, left, right are set.
        Leaf node:     value (predicted class) is set, children are None.
    """
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        samples=0,
        impurity=0.0,
        class_counts=None,
    ):
        """
        Constructor for Node class
        with the attributes and hyperparameters.
        """
        raise  NotImplementedError

    def is_leaf(self):
        """
        Check whether this node is a leaf node.
        Return True if the node has no children; otherwise, return False.
        """
        # TODO
        pass


class DecisionTree:

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        # Hyperparameters
        # criterion_type ="gini or entropy"
        # self.criterion = criterion
        # self.max_depth = max_depth
        # self.min_samples_split = min_samples_split
        # self.min_samples_leaf = min_samples_leaf
        # self.max_features = max_features
        #self.random_state = random_state
        #
        # # Learned attributes
        # self.root_ = None
        # self.n_features_ = None
        # self.n_classes_ = None
        # self.classes_ = None
        # self.n_nodes_ = None
        # self.depth_ = None
        # self.feature_importances_ = None  

        # TODO
        pass

    # =========================
    # Training
    # =========================
    def _weighted_entropy(self, y_left, y_right):
        """
        Calculate the weighted entropy after splitting a node into two children.

        Input:
            y_left: Target labels belonging to the left child.
            y_right: Target labels belonging to the right child.

        Steps:
            1. Calculate the total number of samples in both children.
            2. Calculate the weight of the left child.
            3. Calculate the weight of the right child.
            4. Calculate the entropy of the left child.
            5. Calculate the entropy of the right child.
            6. Calculate the weighted entropy:
                   H_weighted =
                       w_left * H_left + w_right * H_right
            7. Return the calculated weighted entropy.

        Output:
            A float representing the weighted entropy of the two child nodes.
        """
        # TODO
        pass

    def _weighted_gini(self, y_left, y_right):
        """
        Calculate the weighted Gini impurity after splitting a node into two children.

        Input:
            y_left: Target labels belonging to the left child.
            y_right: Target labels belonging to the right child.

        Steps:
            1. Calculate the total number of samples in both children.
            2. Calculate the weight of the left child.
            3. Calculate the weight of the right child.
            4. Calculate the Gini impurity of the left child.
            5. Calculate the Gini impurity of the right child.
            6. Calculate the weighted Gini impurity:
                   Gini_weighted = w_left * Gini_left + w_right * Gini_right
            7. Return the calculated weighted Gini impurity.

        Output:
            A float representing the weighted Gini impurity of the two child nodes.
        """
        # TODO
        pass
    def _entropy(self, y):
        """
        Calculate the entropy of the given target labels.

        Input:
            y: Target labels belonging to a single node.

        Steps:
            1. Find the unique classes in y.
            2. Count the occurrences of each class.
            3. Calculate the probability of each class.
            4. Calculate entropy using:
                   H(Y) = -Σ p_i * log2(p_i)
            5. Return the calculated entropy.

        Output:
            A float representing the entropy of y.
        """
        # TODO
        pass

    def _gini(self, y):
        """
        Calculate the Gini impurity of the given target labels.

        Input:
            y: Target labels belonging to a single node.

        Steps:
            1. Find the unique classes in y.
            2. Count the occurrences of each class.
            3. Calculate the probability of each class.
            4. Calculate Gini impurity using:
                   Gini(Y) = 1 - Σ p_i^2
            5. Return the calculated Gini impurity.

        Output:
            A float representing the Gini impurity of y.
        """
        # TODO
        pass

    def _information_gain(self, y, y_left, y_right):
        """
        Calculate the information gain produced by a split.

        Input:
            y: Target labels of the parent node.
            y_left: Target labels of the left child.
            y_right: Target labels of the right child.

        Steps:
            1. Calculate the impurity of the parent node according to
               the selected criterion.
            2. Calculate the weighted impurity of the two child nodes
               using the selected criterion.
            3. Calculate information gain:
                   Information Gain =
                       Parent Impurity - Weighted Child Impurity
            4. Return the calculated information gain.

        Output:
            A float representing the information gain of the split.
        """
        # TODO
        pass

    def fit(self, X, y):
        """
        Train the Decision Tree using the given training data.

        Input:
            X:
                Training features.
                Shape:
                    (n_samples, n_features)

            y:
                Target labels corresponding to the samples in X.
                Shape:
                    (n_samples,)

        How it works:
            1. Validate or prepare the input data if required.
            2. Store the number of features in the training data.
            3. Determine the unique target classes.
            4. Store the number of classes.
            5. Build the decision tree recursively starting from the root.
            6. Store the resulting root node in self.root_.
            7. Calculate any additional learned attributes required by the
               Decision Tree implementation.

        Output:
            Returns self so that the model can be used in a scikit-learn-like
            manner:

                tree.fit(X, y).predict(X)

        Notes:
            The main recursive tree-construction logic should be implemented
            inside _build_tree().
        """
        # TODO
        pass

    def _build_tree(self, X, y, depth):
        """
        Recursively build a subtree from the given training samples.

        Input:
            X:
                Feature matrix containing the samples that belong to the
                current node.
                Shape:
                    (n_samples, n_features)

            y:
                Target labels corresponding to X.
                Shape:
                    (n_samples,)

            depth:
                Current depth of the node being constructed.
                The root node starts at depth 0.

        How it works:
            1. Check whether the current node should become a leaf.
            2. If a stopping condition is satisfied, create and return
               a leaf node.
            3. Find the best feature and threshold for splitting the
               current samples.
            4. Split X and y into left and right child subsets.
            5. Recursively build the left subtree.
            6. Recursively build the right subtree.
            7. Create an internal Node containing the selected feature,
               threshold, and child nodes.
            8. Return the newly created node.

        Output:
            Returns a Node object representing the root of the subtree
            constructed from the given samples.

            The returned node can be either:
                - a leaf node, or
                - an internal node with left and right children.

        Important:
            This function is recursive. Each recursive call operates on
            a smaller subset of the training data and a greater tree depth.

            The complete implementation should respect the Decision Tree
            hyperparameters such as:
                - max_depth
                - min_samples_split
                - min_samples_leaf
                - criterion
        """
        # 1. Check whether this node should become a leaf.
        # TODO: implement

        # 2. Find the best feature and threshold.
        # TODO: implement

        # 3. Split the dataset using the selected feature and threshold.
        # TODO: implement

        # 4. Build the left subtree.
        # left = self._build_tree(X_left, y_left, depth + 1)

        # 5. Build the right subtree.
        # right = self._build_tree(X_right, y_right, depth + 1)

        # 6. Create the current internal node.
        # TODO: implement
        pass

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold for splitting the current node.

        Input:
            X:
                Feature matrix of the current node.

                Shape:
                    (n_samples, n_features)

                Each row represents one sample and each column represents
                one feature.

            y:
                Target labels corresponding to the samples in X.

                Shape:
                    (n_samples,)

        How it works:
            1. Consider the features that can be used for splitting.

            2. For each feature, determine the candidate thresholds that
               can be used to divide the samples into two groups.

            3. For each candidate threshold:
                   - Split the samples into a left and a right child.
                   - Make sure the resulting split satisfies the required
                     constraints such as min_samples_leaf.
                   - Calculate the quality of the split using the selected
                     criterion.
                   - Measure the improvement produced by the split.

            4. Keep track of the split that produces the best result.

            5. After all candidate features and thresholds have been
               considered, return the best split.

        Important:
            The quality of a split depends on the Decision Tree criterion.

            For example:
                - "entropy" uses entropy-based information gain.
                - "gini" uses Gini impurity reduction.

            A good implementation should not assume that the target labels
            are only 0 and 1. The method should work with multiple classes.

        Output:
            Returns information describing the best split.

            Expected information includes:
                - best feature index
                - best threshold

            If no valid split can be found, the method should return a
            representation that allows _build_tree() to handle this case
            appropriately.

        Notes for implementation:
            - Do not modify X or y inside this method.
            - Be careful when multiple samples have the same feature value.
            - A candidate split must not create invalid child nodes.
            - The method should consider the effect of min_samples_leaf.
        """

        # TODO
        pass

    def _split(self, X, y, feature_index, threshold):
        """
        Split the dataset into left and right child nodes using a
        feature and threshold.

        Input:
            X:
                Feature matrix.

                Shape:
                    (n_samples, n_features)

            y:
                Target labels corresponding to X.

                Shape:
                    (n_samples,)

            feature_index:
                Index of the feature used for splitting.

                Example:
                    feature_index = 2

                means that the third feature of X is used.

            threshold:
                Value used to divide the samples into two groups.

                Samples are assigned to the left or right child according
                to their value for the selected feature.

        How it works:
            1. Select the feature specified by feature_index.

            2. Compare the value of this feature for every sample with
               the given threshold.

            3. Determine which samples belong to the left child.

            4. Determine which samples belong to the right child.

            5. Use the same sample selection to split both X and y.
               This is important because the feature rows and their
               corresponding target labels must remain aligned.

            6. Return the four resulting arrays.

        Output:
            Returns:

                X_left:
                    Feature matrix containing samples assigned to the
                    left child.

                y_left:
                    Target labels corresponding to X_left.

                X_right:
                    Feature matrix containing samples assigned to the
                    right child.

                y_right:
                    Target labels corresponding to X_right.

            The following relationship should hold:

                len(y_left) + len(y_right) == len(y)

            and:

                len(X_left) == len(y_left)
                len(X_right) == len(y_right)

        Important:
            The splitting operation should not change the order of features
            inside X.

            It should also preserve the correspondence between each sample
            and its target label.
        """

        # TODO
        pass

    def _get_candidate_thresholds(self, feature_values):
        """
        Generate candidate thresholds for a single feature.

        Input:
            feature_values:
                Values of one feature for all samples in the current node.

                Example:
                    [1, 2, 3, 4, 5]

        How it works:
            1. Consider the values of the selected feature.
            2. Identify the distinct values.
            3. Sort the distinct values.
            4. Consider the boundaries between consecutive values.
            5. Generate a threshold between each pair of consecutive
               distinct values.
            6. Return the generated thresholds.

        Example:

            feature_values = [1, 2, 3, 4]

            Sorted unique values:
                [1, 2, 3, 4]

            Candidate thresholds:
                (1 + 2) / 2 = 1.5
                (2 + 3) / 2 = 2.5
                (3 + 4) / 2 = 3.5

            Result:
                [1.5, 2.5, 3.5]

        Why do we use midpoints?

            A threshold only needs to separate two consecutive distinct
            feature values.

            For example, thresholds 1.2 and 1.8 produce exactly the same
            partition when the feature values are 1 and 2:

                value <= 1.2    -> {1}
                value > 1.2     -> {2}

                value <= 1.8    -> {1}
                value > 1.8     -> {2}

            Therefore, testing the midpoint is sufficient.

        Important:
            Duplicate feature values should not produce duplicate
            thresholds.

        Output:
            A collection of candidate threshold values that can be used
            by _find_best_split() to evaluate possible splits.
        """

        # TODO
        pass

    def _calculate_impurity(self, y):
        """
        Calculate the impurity of a node according to the selected criterion.

        Input:
            y:
                Target labels of the samples belonging to the current node.

                Shape:
                    (n_samples,)

                Example:
                    [0, 0, 1, 1]

        How it works:
            1. Check which criterion was selected when the DecisionTree
               object was created.

            2. If the criterion is "entropy", calculate the entropy of y.

            3. If the criterion is "gini", calculate the Gini impurity of y.

            4. Return the calculated impurity.

        Output:
            A float representing the impurity of the given node.

            For example:
                criterion = "entropy"
                y = [0, 0, 1, 1]

            returns:
                1.0

        Notes:
            This method should use the existing _entropy() and _gini()
            methods instead of duplicating their mathematical formulas.

            The method should support all valid target labels and should
            not assume that the classes are only 0 and 1.
        """

        # TODO
        pass

    def _calculate_information_gain(self, parent_y, left_y, right_y):
        """
        Calculate the information gain produced by a split.

        Input:
            parent_y:
                Target labels of the parent node before splitting.

            left_y:
                Target labels of the left child after splitting.

            right_y:
                Target labels of the right child after splitting.

        How it works:
            1. Calculate the impurity of the parent node using the
               selected criterion.

            2. Calculate the impurity of the left child.

            3. Calculate the impurity of the right child.

            4. Calculate the number of samples in each child.

            5. Calculate the weight of each child:

                   left_weight =
                       len(left_y) / len(parent_y)

                   right_weight =
                       len(right_y) / len(parent_y)

            6. Calculate the weighted impurity of the children:

                   weighted_child_impurity =
                       left_weight * left_impurity
                       +
                       right_weight * right_impurity

            7. Calculate information gain:

                   information_gain =
                       parent_impurity
                       -
                       weighted_child_impurity

            8. Return the information gain.

        Output:
            A float representing the reduction in impurity caused by
            the split.

            A larger information gain means that the split produces
            a greater reduction in impurity.

        Important:
            The child impurities must be weighted according to their
            number of samples.

            Do not calculate the child impurity as a simple average:

                (left_impurity + right_impurity) / 2

            unless the two children contain the same number of samples.

            The selected criterion must be respected.
        """

        # TODO
        pass

    def _is_stopping_condition(self, X, y, depth):
        """
        Check whether the current node should become a leaf.

        Input:
            X: Feature matrix of the current node.
            y: Target labels of the current node.
            depth: Current depth of the node.

        How it works:
            Check whether the tree should stop splitting the current node.
            Possible conditions include:
                - all samples have the same class
                - maximum depth has been reached
                - there are not enough samples to split

        Output:
            True  -> stop splitting and create a leaf.
            False -> continue splitting the node.
        """

        # TODO
        pass

    def _create_leaf(self, y):
        """
        Create a leaf node using the target labels of the current node.

        Input:
            y: Target labels of the samples in the current node.

        How it works:
            Determine the class that should be predicted by this leaf.
            The leaf should also store any other information required by
            the Node structure.

        Output:
            A Node object representing the leaf.
        """

        # TODO
        pass
    def _predict_one(self, x):
        """
        Predict the class of a single sample.

        Input:
            x:
                Feature values of one sample.

        How it works:
            1. Start from the root of the tree.
            2. Check whether the current node is a leaf.
            3. If it is a leaf, return its predicted class.
            4. Otherwise, compare the sample's feature value with
               the node's threshold.
            5. Move to the left or right child.
            6. Continue until a leaf is reached.

        Output:
            The predicted class for the given sample.
        """

        # TODO
        pass
    def predict(self, X):
        """
        Predict the class of each sample in X.

        Input:
            X:
                Feature matrix containing the samples to predict.

        How it works:
            1. Apply _predict_one() to each sample.
            2. Collect the predicted classes.
            3. Return all predictions.

        Output:
            An array containing one predicted class for each sample.
        """

        # TODO
        pass
