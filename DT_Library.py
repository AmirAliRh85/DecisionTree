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
        if (self.left == None and self.right == None) :
            return True
        return False
    
    def log(self) :
        print(self.feature_index ,
        self.threshold ,
        self.left ,
        self.right ,
        self.value ,
        self.samples ,
        self.impurity ,
        self.class_counts , sep=" ")



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
        #
        # # Learned attributes
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
        y_values , y_counts = np.unique(y , return_counts=True)
        self.classes_ = y_values
        self.classes_count_ = y_counts
        self.n_classes_ = len(y_values)
        self.n_features_ = X.shape[1]
        self.n_nodes_ = 1
        self.depth_ = 0
        self.feature_importances_ = np.zeros(shape=(X.shape[1],))

        if (self._is_stopping_condition(X , y , 0)) :
            self.root_ = self._create_leaf(y)
            return self
        
        best_feature_index , best_threshold = self._find_best_split(X , y)
        self.root_ = Node(feature_index=best_feature_index  , threshold=best_threshold  , samples=X.shape[0] , impurity = self._calculate_impurity(y))
        # self.feature_importances_ = np.append(self.feature_importances_ , best_feature_index)
        
        X_left , y_left , X_right , y_right = self._split(X , y , best_feature_index , best_threshold)
        self.feature_importances_[best_feature_index] = self._entropy(y)
        # print("entropy" , self._entropy(y))
        # print("gini" , self._gini(y))
        # print("weighted entropy" , self._weighted_entropy(y_left , y_right))
        # print("weighted gini" , self._weighted_gini(y_left , y_right))
        # print("impurity" , self._calculate_impurity(y))
        # print("calc information gain" , self._calculate_information_gain(y , y_left , y_right))
        # print("information gain" , self._information_gain(y , y_left , y_right))

        self.root_.left = self._build_tree(X_left , y_left , 1)
        self.root_.right = self._build_tree(X_right , y_right , 1)
        
        return self


    def _build_tree(self , X : np.ndarray , y : np.ndarray , depth):
        self.n_nodes_ += 1
        self.depth_ = max(depth , self.depth_)

        if (self._is_stopping_condition(X , y , depth)) :
            return self._create_leaf(y)
        
        best_feature_index , best_threshold = self._find_best_split(X , y)
        node = Node(feature_index=best_feature_index , threshold=best_threshold  , samples=X.shape[0] , impurity = self._calculate_impurity(y))
        # self.feature_importances_ = np.append(self.feature_importances_ , best_feature_index)

        X_left , y_left , X_right , y_right = self._split(X , y , best_feature_index , best_threshold)
        self.feature_importances_[best_feature_index] = self._entropy(y)
        
        node.left = self._build_tree(X_left , y_left , depth + 1)
        node.right = self._build_tree(X_right , y_right , depth + 1)

        return node
    

    def _find_best_split(self , X : np.ndarray , y : np.ndarray):
        if (self.min_samples_leaf > X.shape[0]) :
            return None

        best_feature_index = 0
        best_threshold = 0
        information_gain = -1

        for j in range(X.shape[1]) :
            current_column = X[: , j]
            candidates = self._get_candidate_thresholds(current_column)
            new_information_gain = -1

            for cand in candidates :
                y_left = np.array([])
                y_right = np.array([])
                for i in range(len(y)) :
                    if (X[i][j] <= cand) :
                        y_left = np.append(y_left , y[i])
                    else :
                        y_right = np.append(y_right , y[i])
                        
                new_information_gain = max(self._information_gain(y , y_left , y_right) , new_information_gain)
                
                if (len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf) :
                    continue

                if (new_information_gain > information_gain) :
                    best_feature_index = j
                    best_threshold = cand

                information_gain = new_information_gain

        # print("information_gain:" , information_gain)           # for debugging
        # print("best_feature_index:" , best_feature_index)       # for debugging
        # print("best_thresold:" , best_threshold)                # for debugging

        if (information_gain < 0) :
            return None
        
        return (best_feature_index , best_threshold)


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
        if ((not self.max_depth == None and depth >= self.max_depth) or X.shape[0] < self.min_samples_split) :
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

        node = Node(value=node_value)
        return node
    




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
        if (self.root_ == None) :
            raise RuntimeError
            return

        prediction = np.ndarray(shape=(X.shape[0],) , dtype=object)

        idx = 0
        for x in X :
            prediction[idx] = self._predict_one(x)
            idx += 1

        return prediction
