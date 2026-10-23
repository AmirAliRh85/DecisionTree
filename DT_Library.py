import math
import numpy as np

# [] _find_best_split should be more optimized in the append section

class Node:
    def __init__(self , feature_index=None , threshold=None , left=None , right=None , value=None , samples=0 , impurity=0.0 , class_counts=None) :
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity
        self.class_counts = class_counts


    def is_leaf(self):
        return self.left is None and self.right is None


class DecisionTree:

    def __init__(self , criterion="gini" , max_depth=None , min_samples_split=2 , min_samples_leaf=1 , max_features=None , random_state=None):
        # Hyperparameters
        # criterion_type ="gini or entropy"
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        # Learned attributes
        self.root_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.classes_count_ = None
        self.n_nodes_ = None
        self.depth_ = None
        self.feature_importances_ = None
  

    def _weighted_entropy(self , y_left : np.ndarray , y_right : np.ndarray) :
        w_left = len(y_left) / (len(y_left) + len(y_right))
        w_right = len(y_right) / (len(y_left) + len(y_right))

        y_left_values , y_left_counts = np.unique(y_left , return_counts=True)
        y_right_values , y_right_counts = np.unique(y_right , return_counts=True)

        y_left_entropy = 0
        y_right_entropy = 0
        
        for i in y_left_counts :
            p = (i / len(y_left))
            if (p != 0) :
                y_left_entropy += p * math.log(p , 2)
            
        for i in y_right_counts :
            p = (i / len(y_right))
            if (p != 0) :
                y_right_entropy += p * math.log(p , 2)

        y_left_entropy *= -1
        y_right_entropy *= -1

        return y_left_entropy * w_left + y_right_entropy * w_right


    def _weighted_gini(self, y_left, y_right) :
        if (len(y_left) == 0 and len(y_right) == 0) :
            return 0
        
        w_left = len(y_left) / (len(y_left) + len(y_right))
        w_right = len(y_right) / (len(y_left) + len(y_right))

        y_left_values , y_left_counts = np.unique(y_left , return_counts=True)
        y_right_values , y_right_counts = np.unique(y_right , return_counts=True)

        y_left_gini = 1
        y_right_gini = 1

        for i in y_left_counts :
            p = (i / len(y_left))
            y_left_gini -= p * p

        for i in y_right_counts :
            p = (i / len(y_right))
            y_right_gini -= p * p

        return y_left_gini * w_left + y_right_gini * w_right
                        

    def _entropy(self, y):
        y_values , y_counts = np.unique(y , return_counts=True)

        y_entropy = 0
        
        for i in y_counts :
            p = (i / len(y))
            if (p != 0) :
                y_entropy += p * math.log(p , 2)
            
        y_entropy *= -1

        return y_entropy


    def _gini(self, y):
        y_values , y_counts = np.unique(y , return_counts=True)

        y_gini = 1

        for i in y_counts :
            p = (i / len(y))
            y_gini -= p * p

        return y_gini


    def _information_gain(self, y, y_left, y_right):
        if (self.criterion == "gini") :
            return self._gini(y) - self._weighted_gini(y_left , y_right)
        if (self.criterion == "entropy") :
            return self._entropy(y) - self._weighted_entropy(y_left , y_right)


    def fit(self , X : np.ndarray , y : np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"X must be numeric; got dtype={X.dtype}. Convert with np.asarray(X, dtype=float) first.")

        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        if X.shape[0] == 0:
            raise ValueError("empty dataset")

        y_values , y_counts = np.unique(y , return_counts=True)
        self.classes_ = y_values
        self.classes_count_ = y_counts
        self.n_classes_ = len(y_values)
        self.n_features_ = X.shape[1]
        self.n_nodes_ = 0
        self.depth_ = 0
        self.feature_importances_ = np.zeros(self.n_features_)

        if (self._is_stopping_condition(X , y , 0)) :
            self.root_ = self._create_leaf(y)
            self._compute_tree_stats()
            return self
        
        self.root_ = self._build_tree(X , y , 0)
        self._compute_tree_stats()
        
        return self


    def _build_tree(self , X : np.ndarray , y : np.ndarray , depth):
        if (self.n_nodes_ is None) :
            self.n_nodes_ = 0
        if (self.depth_ is None) :
            self.depth_ = 0
        
        self.n_nodes_ += 1
        self.depth_ = max(depth , self.depth_)

        if (self._is_stopping_condition(X , y , depth)) :
            return self._create_leaf(y)
        
        split = self._find_best_split(X , y)
        if (split is None) :
            return self._create_leaf(y)

        best_feature_index = split[0]
        best_threshold = split[1]

        y_values, y_counts = np.unique(y, return_counts=True)
        majority = y_values[int(np.argmax(y_counts))]
        node = Node(
            feature_index = best_feature_index,
            threshold     = best_threshold,
            samples       = X.shape[0] if type(X) is np.ndarray else None,
            impurity      = self._calculate_impurity(y),
            value         = majority,
            class_counts  = {v: int(c) for v, c in zip(y_values, y_counts)})

        X_left , y_left , X_right , y_right = self._split(X , y , best_feature_index , best_threshold)

        node.left = self._build_tree(X_left , y_left , depth + 1)
        node.right = self._build_tree(X_right , y_right , depth + 1)

        return node
    

    # def _find_best_split(self , X : np.ndarray , y : np.ndarray):
    #     if (self.min_samples_leaf > X.shape[0]) :
    #         return None

    #     best_feature_index = 0
    #     best_threshold = 0
    #     information_gain = -1

    #     for j in range(X.shape[1]) :
    #         current_column = X[: , j]
    #         candidates = self._get_candidate_thresholds(current_column)
    #         new_information_gain = -1

    #         for cand in candidates :
    #             y_left_mask = X[: , j] <= cand
    #             y_left = y[y_left_mask]
    #             y_right = y[~y_left_mask]
    #             new_information_gain = max(self._information_gain(y , y_left , y_right) , new_information_gain)
                
    #             if (len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf) :
    #                 continue

    #             if (new_information_gain > information_gain) :
    #                 best_feature_index = j
    #                 best_threshold = cand

    #             information_gain = new_information_gain


    #     if (information_gain < 0) :
    #         return None
        
    #     return (best_feature_index , best_threshold)
    
    
    # optimized version
    def _impurity_from_counts(self, counts):
        n = counts.sum()
        if n == 0:
            return 0.0
        
        p = counts / n
        if self.criterion == "gini":
            return 1.0 - float(np.dot(p, p))
        p_pos = p[p > 0]

        return float(-np.dot(p_pos , np.log2(p_pos)))


    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        min_leaf = self.min_samples_leaf

        if n < self.min_samples_split or n < 2 * min_leaf:
            return None

        _, y_enc = np.unique(y, return_inverse=True)
        n_classes = int(y_enc.max()) + 1
        parent_counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
        parent_impurity = self._impurity_from_counts(parent_counts)

        left_n = np.arange(1 , n , dtype=np.float64)
        right_n = n - left_n
        leaf_ok = (left_n >= min_leaf) & (right_n >= min_leaf)
        if not leaf_ok.any():
            return None

        is_gini = (self.criterion == "gini")
        row_idx = np.arange(n)

        # best_gain  = -100000000
        best_gain  = -np.inf
        best_split = None

        for j in range(X.shape[1]):
            col = X[: , j]
            order = np.argsort(col)
            col_sorted = col[order]
            y_sorted = y_enc[order]

            oh = np.zeros((n, n_classes) , dtype=np.float64)
            oh[row_idx , y_sorted] = 1.0
            left_counts  = np.cumsum(oh, axis=0)[ : -1]
            right_counts = parent_counts - left_counts

            valid = (col_sorted[:-1] < col_sorted[1:]) & leaf_ok
            if not valid.any():
                continue

            if is_gini:
                left_sq  = (left_counts  * left_counts ).sum(axis=1)
                right_sq = (right_counts * right_counts).sum(axis=1)
                left_imp  = 1.0 - left_sq  / (left_n  * left_n)
                right_imp = 1.0 - right_sq / (right_n * right_n)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    lp = np.where(left_counts > 0 , left_counts / left_n[: , None] ,  1.0)
                    rp = np.where(right_counts > 0 , right_counts / right_n[: , None] , 1.0)
                    left_imp  = -(lp * np.log2(lp)).sum(axis=1)
                    right_imp = -(rp * np.log2(rp)).sum(axis=1)

            weighted = (left_n / n) * left_imp + (right_n / n) * right_imp
            gains = parent_impurity - weighted
            gains[~valid] = -np.inf

            idx = int(np.argmax(gains))
            if gains[idx] > best_gain:
                best_gain  = gains[idx]
                threshold  = 0.5 * (col_sorted[idx] + col_sorted[idx + 1])
                best_split = (j , float(threshold))

        return best_split


    def _split(self, X, y, feature_index, threshold) :
        left_mask = X[: , feature_index] <= threshold
        right_mask = ~left_mask

        X_left = X[left_mask]
        X_right = X[right_mask]

        y_left = y[left_mask]
        y_right = y[right_mask]
        
        return (X_left , y_left , X_right , y_right)


    def _get_candidate_thresholds(self, feature_values : np.ndarray):
        sorted_feature_values = np.sort(np.unique(feature_values))
        candidate_threshhold = np.ndarray(shape=(len(sorted_feature_values) - 1,))

        if (len(sorted_feature_values) <= 1) :
            return candidate_threshhold
        else :
            for i in range(0 , len(sorted_feature_values) - 1) :
                candidate_threshhold[i] = (sorted_feature_values[i] + sorted_feature_values[i + 1]) / 2

        return candidate_threshhold

    def _calculate_impurity(self, y):
        if (self.criterion == "gini") :
            return self._gini(y)
        elif (self.criterion == "entropy") :
            return self._entropy(y)
                

    def _calculate_information_gain(self, parent_y, left_y, right_y):
        parent_impurity = self._calculate_impurity(parent_y)
        left_child_impurity = self._calculate_impurity(left_y)
        right_child_impurity = self._calculate_impurity(right_y)
        
        if (len(left_y) == 0) :
            left_child_impurity = 0
        else :
            left_child_impurity = left_child_impurity * (len(left_y) / len(parent_y))
        
        if (len(right_y) == 0) :
            right_child_impurity = 0
        else :
            right_child_impurity = right_child_impurity * (len(right_y) / len(parent_y))
        
        return parent_impurity - left_child_impurity - right_child_impurity

    def _is_stopping_condition(self , X : np.ndarray , y : np.ndarray , depth):
        if ((not self.max_depth == None and not self.max_depth == np.inf and not self.max_depth == float("inf") and depth >= self.max_depth) or X.shape[0] < self.min_samples_split) :
            return True
        
        y_values , y_counts = np.unique(y , return_counts=True)
        if (len(y_values) == 1) :
            return True
        
        return False

    def _create_leaf(self , y) :
        y_values , y_counts = np.unique(y , return_counts=True)

        node_value = 0
        class_count_max = -1
        idx = 0
        for i in y_counts :
            if (i > class_count_max) :
                class_count_max = i
                node_value = y_values[idx]
            idx += 1

        node = Node(value=node_value , impurity=self._calculate_impurity(y) , samples=len(y) , class_counts={v: int(c) for v, c in zip(y_values, y_counts)})
        return node
    

    def _compute_tree_stats(self) :
        if self.root_ is None :
            self.n_nodes_ = 0
            self.depth_ = 0
            self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
            return

        importances = np.zeros(self.n_features_, dtype=float)

        def __computer_tree_stats(node, depth):
            if not node.is_leaf():
                n = node.samples
                n_left  = node.left.samples
                n_right = node.right.samples

                gain = node.impurity - (n_left  / n) * node.left.impurity - (n_right / n) * node.right.impurity
                importances[node.feature_index] += (n / self.root_.samples) * gain

                size_left , depth_left = __computer_tree_stats(node.left ,  depth + 1)
                size_right , depth_right = __computer_tree_stats(node.right , depth + 1)
                return size_left + size_right + 1, max(depth_left, depth_right)

            return 1 , depth

        total_nodes , max_depth = __computer_tree_stats(self.root_ , 0)

        self.n_nodes_ = total_nodes
        self.depth_   = max_depth

        total = importances.sum()
        if total > 0:
            importances /= total    # normalization
        self.feature_importances_ = importances



    def _predict_one(self , x):
        node = self.root_
        while (not node.is_leaf()) :
            threshold = node.threshold
            feature_index = node.feature_index
            if (threshold >= x[feature_index]) :
                node = node.left
            else :
                node = node.right

        return node.value

    def predict(self , X):
        if (self.root_ is None) :
            raise RuntimeError
        
        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"X has shape {X.shape}; expected (n, {self.n_features_})")
        
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"X is {X.dtype}; convert with np.asarray(X, dtype=float). Some cells are strings (e.g. {X.flat[0]!r}).")

        prediction = np.ndarray(shape=(X.shape[0],) , dtype=object)

        idx = 0
        for x in X :
            prediction[idx] = self._predict_one(x)
            idx += 1

        return prediction
