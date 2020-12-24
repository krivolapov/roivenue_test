import numpy as np
import random
import datetime
#import matplotlib.pyplot as plt
from datetime import timedelta
import copy
import json

#DEBUG
import os
import psutil

from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

random.seed(42)

def timedelta_to_days(timedelta):
    return timedelta / (1000 * 60*60*24)

def days_to_timedelta(days):
    return days * 1000 * 60*60*24

class ModelData:
    pass

class Settings:
    pass

class ConvProbModel:
    
    def __init__(self, settings = None, model_data = None):
        self._settings = settings
        self._model_data = model_data

    

    def load_settings_from_json(self, settings_json):
        def deserialize_settings(d):
            s = Settings()
            for k in d:
                s.__dict__[k] = d[k]
            return s
        
        self._settings = json.loads(settings_json, object_hook = deserialize_settings)
        
    def _build_category_one_hot_mappings(self, paths):
        category_columns = self._settings.columns.category_columns
        one_hot_counters = dict(map(lambda col: (col, {}), category_columns))
        one_hot_mappings = {}

        for path in paths:
            for session in path:
                for category in category_columns:
                    value = one_hot_counters[category].get(session[category], 0)
                    one_hot_counters[category][session[category]] = value + 1

        for category in category_columns:
            items = one_hot_counters[category].items()
            limit = self._settings.hyperparameters.category_columns_limit
            limited_items = (list(sorted(items, key = lambda item: item[1], reverse = True))[0:limit])
            one_hot_mappings[category] = {}
            for i, item in enumerate(limited_items):
                one_hot_mappings[category][item[0]] = i
        
        self._model_data.one_hot_mappings = one_hot_mappings
    
    def _init_extractors(self):
        columns = self._settings.columns
        feature_metric_columns = list(filter(lambda col: not col in columns.columns_removed_from_training, columns.metric_columns))
        
        metric_extractors = list(map(lambda arg: lambda path, sess, index: sess[arg], feature_metric_columns))
        log_metric_extractors = list(map(lambda arg: lambda path, sess, index: np.log(sess[arg] + 1), feature_metric_columns))
    
        extractors = []

        extractors.extend(metric_extractors)
        extractors.extend(log_metric_extractors)

        category_extractors = []
        
        #closure for category
        def get_one_hot_match(column, category):
            return lambda path, sess, index: 1 if sess[column] == category else 0 

        for category_column in self._model_data.one_hot_mappings.items():
            for category in category_column[1]:
                category_extractors.append(get_one_hot_match(category_column[0], category))
            #TODO add one-hot mapping for other column

        extractors.extend(category_extractors)
        
        self._feature_extractor_abs_time_index = len(extractors)
        extractors.append(lambda path, sess, index: 0 if index == 0 else timedelta_to_days(sess["timestamp"] - path[0]["timestamp"]))
        self._feature_extractor_rel_time_index = len(extractors)
        extractors.append(lambda path, sess, index: 0 if index == 0 else timedelta_to_days(sess["timestamp"] - path[index - 1]["timestamp"]))
                

        self._feature_extractors = extractors
        
        self._label_extractor = lambda sess: 1 if sess["conversions"] > 0 else 0 
    
    def get_vectors(self, paths, limit_touchpoints = None):
        
        T_m = self._settings.hyperparameters.T_m
        if limit_touchpoints is None:
            limit_touchpoints = T_m
        n_features = len(self._feature_extractors)
        time_step = self._settings.hyperparameters.time_step_padding
        
        x = np.zeros((len(paths), T_m, n_features))
        y = np.zeros((len(paths), 1))

        for path_index, path in enumerate(paths):
            
            #print(path) #debug
            
            #needed?
            limited_touchpoints = len(path) if len(path) < limit_touchpoints else limit_touchpoints;

            features = []
            importances = []
            startDate = path[0]["timestamp"]
            lastDate = startDate
            #enumerate touchpoints
            for index, session in enumerate(path):

                #limiting numer of touchpoints to see effect of only touchpoints before limit
                if index >= limit_touchpoints:
                    break

                #pad more distant events by empty space
                while timedelta_to_days(session["timestamp"] - lastDate) > time_step:
                    lastDate += days_to_timedelta(time_step)
                    gap_vect = np.zeros(n_features) #TODO use n_features
                    gap_vect[self._feature_extractor_rel_time_index] = time_step
                    gap_vect[self._feature_extractor_abs_time_index] = timedelta_to_days(lastDate - startDate)
                    features.append(gap_vect)
                    importances.append(1)
                    #print("s " + str(lastDate))


                v = np.fromiter(map(lambda ex: ex(path, session, index), self._feature_extractors), dtype=float)
                v[self._feature_extractor_rel_time_index] = timedelta_to_days(session["timestamp"] - lastDate)
                features.append(v)

                #if there is a conversion-> impotance 10000, 100 for each session, 10 otherwise
                #TODO make an extractor for this
                importance = v[0] * 10000 + 10 * v[1] + 1
                importances.append(importance)
                lastDate = session["timestamp"]

            #reduce the length of sequence by merging the least important events
            #compressed_limit_touchpoints = limit_touchpoints

            while len(features) > T_m:
                min_index = 0
                min_value = importances[0] + importances[1]

                for i in range(1, len(importances) - 2): #-2 is because we allways keep last touch
                    if min_value > importances[i] + importances[i + 1]:
                        min_value = importances[i] + importances[i + 1]
                        min_index = i

                features[min_index] = features[min_index] + features[min_index + 1]
                features.pop(min_index + 1)

                #TODO can be merged into one array (when importance is a feature)
                importances[min_index] = importances[min_index] + importances[min_index + 1]
                importances.pop(min_index + 1)

            features_mat = np.array(features)

            #old version: x[path_index, 0:features_mat.shape[0], :] = features_mat
            x[path_index, -features_mat.shape[0]:, :] = features_mat
            y[path_index, 0] = max(map(self._label_extractor, path))

        return x, y
        
    def get_train_and_test_set(self, paths):
        #train vs. test split
        random.shuffle(paths)
        
        split_ratio = self._settings.hyperparameters.train_test_split_ratio
        train_paths = paths[(len(paths) // split_ratio):]
        test_paths = paths[0:(len(paths) // split_ratio)]
        
        
        all_conversion_paths = list(filter(lambda p: any(t[self._settings.columns.conversion_column] > 0 for t in p), paths))
        #all_nonconversion_paths = list(filter(lambda p: not any(t["conversions"] > 0 for t in p), paths))

        #TODO do not use balancing.
        #use class_weights instead: 
        #https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras

        #balancing training set
        conversion_train_paths = list(filter(lambda p: any(t[self._settings.columns.conversion_column] > 0 for t in p), train_paths))
        nonconversion_train_paths = list(filter(lambda p: not any(t[self._settings.columns.conversion_column] > 0 for t in p), train_paths))

        balancing_modifier = self._settings.hyperparameters.non_conversion_train_set_balancing
        if len(conversion_train_paths) * balancing_modifier < len(nonconversion_train_paths):
            balanced_non_conversion_paths = nonconversion_train_paths[0:len(conversion_train_paths) * balancing_modifier];
            train_paths = conversion_train_paths + balanced_non_conversion_paths
            random.shuffle(train_paths)
        
        #limit number of train paths (performance reasons)
        if len(train_paths) > self._settings.hyperparameters.max_train_paths:
            train_paths = train_paths[0:self._settings.hyperparameters.max_train_paths]
        
        print(f'Total conversion paths: {len(all_conversion_paths)}, train conversion paths {len(conversion_train_paths)}, total train paths: {len(train_paths)}')
        
        return train_paths, test_paths
    
    def get_channel(self, touchpoint):
        return '/'.join(map(lambda c: touchpoint[c], self._settings.columns.channel_columns))
    
    def train_nn_model(self, x_train, y_train):
        hp = self._settings.hyperparameters
        
        model = Sequential()
        #TODO GRU or LSTM?
        model.add(layers.GRU(hp.nn_rnn_size, input_shape = (x_train.shape[1], x_train.shape[2])))
        model.add(layers.Dense(hp.nn_dense1_size, activation = 'relu'))
        model.add(layers.Dense(1, activation = 'sigmoid'))

        model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        model.fit(x_train, y_train, batch_size = hp.batch_size, epochs = hp.epochs)
        
        self._trained_model = model
    
    def evaluate_nn_model(self, x_test, y_test):
        y_test_pred_p = self._trained_model.predict(x_test).reshape([x_test.shape[0]])
        y_pred = np.where(y_test_pred_p > 0.5, 1, 0)

        self._confusion_matrix = str(confusion_matrix(y_test, y_pred))
        print("confusion matrix(below):")
        print(self._confusion_matrix)  
        print("classification report(below):")
        print(classification_report(y_test, y_pred,digits=4))
    
    def get_confusion_matrix(self):
        return self._confusion_matrix

    def train_model(self, paths):
        self._model_data = ModelData()
        self._build_category_one_hot_mappings(paths)
        self._init_extractors()

        random.seed(self._settings.hyperparameters.random_seed)
        
        train_paths, test_paths = self.get_train_and_test_set(paths)

        #DEBUG
        #print(f'Memory {psutil.Process(os.getpid().memory_info().rss)}')

        print(f'Retrieving features ({len(train_paths)} paths)...')
        x_train, y_train = self.get_vectors(train_paths, None)
        x_test, y_test = self.get_vectors(test_paths, None)
        
        self.train_nn_model(x_train, y_train)
        
        print(f'Evaluating model ({len(test_paths)} paths)...')
        self.evaluate_nn_model(x_test, y_test)
       
    def get_model(self):
        return self._trained_model
    
    def export_model(self):
        data = {}
        data["one_hot_mappings"] = self._model_data.one_hot_mappings
        data["model_json"] = self._trained_model.to_json()
        weights = self._trained_model.get_weights()
        data["model_weights"] = list(map(lambda l: l.tolist(), weights))
        data["settings"] = self._settings
        
        return data 

    def export_model_to_json(self):
        data = self.export_model()

        def serialize(o):
            try:
                return o.__dict__
            except:
                print("Serialization error on:")
                print(o)
                return {}
        return json.dumps(data, default = serialize)
      
    def load_model(self, data):
        self._model_data = Settings()
        self._model_data.one_hot_mappings = data["one_hot_mappings"]
        self._init_extractors()
      
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = model_from_json(data["model_json"])
        
        model.set_weights(list(map(lambda l: np.array(l, dtype=float), data["model_weights"])))
        self._trained_model = model
        
    def load_model_from_json(self, json_data):
        data = json.loads(json_data)
        settings_json = json.dumps(data["settings"])
        self.load_settings_from_json(settings_json)
        self.load_model(data)
    
    def set_model(self, model_data, trained_model):
        self._trained_model = trained_model
        self._model_data = model_data
        self._init_extractors()
    
    def get_x_for_path(self, path):
        x = np.zeros((len(path), self._settings.hyperparameters.T_m, len(self._feature_extractors)))
                
        for i in range(0, len(path)):
            xs, ys = self.get_vectors([path], i + 1)
            x[i, :, :] = xs
        return x
    
    def get_conversion_probabilities(self, x):
        return self._trained_model.predict(x).reshape([x.shape[0]])
    
    def get_touchpoint_scores(self, path, scoring = None):
        res = self.get_touchpoint_scores_raw(path, scoring)
        return res[0]
    
    def get_touchpoint_scores_raw(self, path, scoring = None):

        x = self.get_x_for_path(path)
        probabilities = self.get_conversion_probabilities(x)
        
        if scoring is None:
            scoring = getattr(self, self._settings.hyperparameters.scoring_function)
        elif isinstance(scoring, str):
            scoring = getattr(self, scoring)
 
        return scoring(probabilities, path)
    
    def normalize_scores(self, scores):
        scores[scores < 0] = 0
        scores_sum = np.sum(scores)
        if scores_sum > 0:
            return scores / scores_sum
        else:
            return np.ones(scores.shape[0]) / scores.shape[0]
    
    def apriori_prob(self, path):
        return self._settings.hyperparameters.apriori_log_prob
    
    def is_organic_touchpoint(self, touchpoint):
        return touchpoint["webSource"] == "(direct)" or touchpoint["webMedium"] == "organic"
    
    def get_nonorganic_compensation(self, scores, path):
        nonorganic_compensation = np.zeros(scores.shape)
        last_nonorganic_index = -1
        for i, touchpoint in enumerate(path):
            if self.is_organic_touchpoint(path[i]):
                if last_nonorganic_index >= 0:
                    t_dist = timedelta_to_days(path[i]["timestamp"] - path[last_nonorganic_index]["timestamp"])
                    influence_coeff = np.exp(-self._settings.hyperparameters.scoring_time_influence_rate * t_dist)
                    compensation = scores[i] * influence_coeff
                    nonorganic_compensation[last_nonorganic_index] = nonorganic_compensation[last_nonorganic_index] + compensation
                    nonorganic_compensation[i] = nonorganic_compensation[i] - compensation
            else:
                last_nonorganic_index = i
        return nonorganic_compensation
    
    def get_decay_coeffs(self, tscore, path):
        decay_coeff = np.zeros(tscore.shape)
        for i in range(1, len(path)):
            t_dist = timedelta_to_days(path[i]["timestamp"] - path[i - 1]["timestamp"])
            decay_coeff[i] = np.exp(-self._settings.hyperparameters.scoring_time_decay_rate * t_dist)
        return decay_coeff
    
    def score_linear (self, ps, path):
        pss = np.zeros(ps.shape)
        pss[1:] = ps[:-1]
        vals = (ps - pss).ravel()
        return (self.normalize_scores(vals), ps)
        
    def score_log (self, ps, path):
        pss = np.zeros(ps.shape)
        
        pss[0] = self.apriori_prob(path)
        pss[1:] = ps[:-1]
        vals = (np.log(ps) - np.log(pss)).ravel()
        return (self.normalize_scores(vals), ps)
        
        
    def score_log_with_time_decay (self, ps, path):
        tscore = np.array(ps)
        
        #cross_influence = self.get_touchpoint_crossinfluence(tscore, path)
       
        decay_coeffs = self.get_decay_coeffs(tscore, path)

        shifted_tscore = np.zeros(tscore.shape)
        shifted_tscore[0] = self.apriori_prob(path)
        shifted_tscore[1:] = (tscore[:-1] * decay_coeffs[1:]) # + cross_influence[:-1]

        vals = np.maximum((np.log(tscore) - np.log(shifted_tscore)).ravel(), 0)
        return (self.normalize_scores(vals), ps, shifted_tscore, decay_coeffs)
    
    
    
    def score_optimized_with_overlaps (self, ps, path):
        tscore = np.array(ps)
        
        #cross_influence = self.get_touchpoint_crossinfluence(tscore, path)
       
        decay_coeffs = self.get_decay_coeffs(tscore, path)

        shifted_tscore = np.zeros(tscore.shape)
        shifted_tscore[0] = self.apriori_prob(path)
        shifted_tscore[1:] = (tscore[:-1] * decay_coeffs[1:])# + cross_influence[:-1]

        
        #lin_diff = np.maximum(tscore - shifted_tscore, 0)
        #log_diff = np.log(tscore) - np.log(shifted_tscore)
        #vals = np.maximum(lin_diff, log_diff * 0.1)
        
        tscore[-1] = 1
        s = 0.05 if self.is_organic_touchpoint(path[0]) else 0.15
            
        vals = np.maximum(tscore - shifted_tscore, np.maximum((np.log10(tscore) - np.log10(shifted_tscore)) * 0.5, 0))
        
        nonorganic_compensation = self.get_nonorganic_compensation(vals, path)
        vals = vals + nonorganic_compensation
        
        return (self.normalize_scores(vals), ps, vals, tscore, shifted_tscore, nonorganic_compensation)
        
    
    