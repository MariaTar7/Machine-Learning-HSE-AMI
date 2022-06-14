import numpy as np
#import pandas as pd
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
 
    
    #не рассматривает пороги, при которых при делении в подмножестве меньше элементов, чем min_samples_leaf
    data = np.vstack((feature_vector, target_vector)).T
    data = data[data[:, 0].argsort()]
    target_cumsum = np.cumsum(data[:, 1])
    u, indices = np.unique(data[:, 0], return_index=True)
    thresholds = (u[1:] + u[:-1])/2
    inx = indices[1:]
    target_size = target_vector.size 
    #new
    if min_samples_leaf is not None:   #new
        inx = inx[inx >= min_samples_leaf] #new
        inx = inx[(target_size - inx) >= min_samples_leaf] #new
    if len(inx) == 0: #new
        return None, None, None, None #new
    cumsum_left = target_cumsum[inx - 1]
    target_sum = np.sum(target_vector) 
    cumsum_right = target_sum - cumsum_left
    p1_l = cumsum_left/inx
    p0_l = 1 - p1_l
    target_size = target_vector.size
   
    
    p1_r = (cumsum_right)/(target_size - inx)
    p0_r = 1 - p1_r
    H_l = 1 - p0_l**2 - p1_l**2  
    H_r = 1 - p0_r**2 - p1_r**2 
    ginis = - inx/target_size * H_l - (target_size - inx)/target_size * H_r
    if u.size == 1:
        return 
    gini_best = np.max(ginis) 
    threshold_best = thresholds[np.argmax(ginis)] 
    return thresholds, ginis, threshold_best, gini_best
    pass


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {'depth': 0}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        
        #было так if np.all(sub_y != sub_y[0]): 
        #НАОБОРОТ ==, ЕСЛИ В САБ У ВСЕ ТАРГЕТЫ РАВНЫ, НЕ НАДО РАЗБИВАТЬ ДАЛЬШЕ
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return 
        if self._max_depth is not None and self._max_depth == node['depth']: #new
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]        
            return
        if (self._min_samples_split is not None) and (sub_y.size < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]        
            return
            
            

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                #print(sub_X[:, feature].shape)
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # было так ratio[key] = current_count / current_click 
                    #ТУТ ЧТО ТО НЕ ТО ДЕЛЕНИЕ НА 0 МБ НАОБОРОТ
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                # было так sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
          
            else:
                raise ValueError
            

            #if len(feature_vector) == 3:
                #continue
            if len(set(feature_vector)) == 1:  
                continue
            

            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf) #new добавила листья
            if threshold == None: #new
                continue #new
            
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))  
                else:
                    raise ValueError

        if feature_best is None:
            
            node["type"] = "terminal"
            # было node["class"] = Counter(sub_y).most_common(1) эта штука возвращает пару
            node["class"] = Counter(sub_y).most_common(1)[0][0]        
            return
        #гуд
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        # при переходе в правое или левое дерево глубина увеличивается на 1
        node["left_child"], node["right_child"] = {'depth': node["depth"] + 1}, {'depth': node["depth"] + 1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        # было так self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
      
        if node['type'] == 'terminal':
            result = node['class']
            return result
        if self._feature_types[node['feature_split']] == "real":
            if x[node['feature_split']] <= node['threshold']:
                #влево
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        elif self._feature_types[node['feature_split']] == "categorical":
        
            if x[node['feature_split']] in node['categories_split']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
           
        else:
            raise ValueError
        pass
    #
    #

    def fit(self, X, y):
        self._tree = {'depth': 1}
        self._fit_node(X, y, self._tree)
        #print(self._tree)
        
        
    def get_params(self, deep = False):
        return {'feature_types': self._feature_types} #'max_depth':self.max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf' : min_samples_leaf}
    
    def set_params(self, **parameters):
        self._max_depth = parameters['max_depth']
        self._min_samples_split = parameters['min_samples_split']
        self._min_samples_leaf = parameters['min_samples_leaf']

        return self


    def predict(self, X):
        predicted = []
        for x in X:
            #print(self._predict_node(x, self._tree))
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    
    