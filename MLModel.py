import json
import pandas as pd
import numpy as np
from striprtf.striprtf import rtf_to_text
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import GridSearchCV, train_test_split,TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,PolynomialFeatures, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression,SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score,make_scorer, roc_auc_score, f1_score
from striprtf.striprtf import rtf_to_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier,SGDRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler


class FeatureGenerationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None, feature_generation=None):
        self.col_n=feature_names
        self.feature_generation=feature_generation
    
    def fit(self, X, y=None):
        return self
    
    def _generate_interactions(self, X):

        # Linear Interactions
        for f1, f2 in self.feature_generation.get("linear_interactions", []):
            interaction = X[f1] * X[f2]
            if self.feature_generation.get("linear_scalar_type") == "robust":
                scaler = RobustScaler()
                interaction = scaler.fit_transform(interaction.values.reshape(-1, 1)).flatten()
            X['linear_interactions']=interaction

        # Polynomial Interactions (Ratios)
        def interaction_terms(X):
            X = np.asarray(X)  # Ensure input is a NumPy array
            return X[:, 2] / X[:, 1]*X[:, 3] / (X[:, 4] + 1e-8) # petal_length / sepal_width
                  # petal_width / species
        if 'polynomial_interactions' in self.feature_generation.keys():
            X['polynomial_interactions']= interaction_terms(X)

        # Explicit Pairwise Interactions (Ratios)
        def pairwise_interactions(X):
            X = np.asarray(X)  # Ensure input is a NumPy array
            return X[:, 1] / X[:, 0]* X[:, 3] / X[:, 0]  # sepal_width / sepal_length
                                                        # petal_width / sepal_length
            
        if 'explicit_pairwise_interactions' in self.feature_generation.keys():
            X['pairwise_interactions']= pairwise_interactions(X)
        
        return X

    def transform(self, X):
        X=pd.DataFrame(X,columns=self.col_n)
        X_transformed = self._generate_interactions(X)
        return X_transformed

class MLModel:
    def __init__(self, json_path):
        self.json_data = self.load_json(json_path)
        self.target, self.train, self.feature_handling, self.feature_generation, self.feature_reduction, self.algorithms = self.parse_json(self.json_data)
        self.df = self.load_dataset(self.json_data['design_state_data']['session_info']['dataset'])
        self.col_n = []
        self.trs = self.create_transformer()
        self.feature_generation=('feature_generation', FeatureGenerationTransformer(feature_generation=self.feature_generation,feature_names=self.col_n))
        self.df_=self.pipe1()
        self.feature_red = self.apply_feature_reduction(self.feature_reduction)
        self.D_Regressor = {
    'RandomForestRegressor': 'RandomForestRegressor',
    'GBTRegressor': 'GradientBoostingRegressor',
    'LinearRegression': 'LinearRegression',
    'LogisticRegression': 'LogisticRegression',
    'RidgeRegression': 'Ridge',
    'LassoRegression': 'Lasso',
    'ElasticNetRegression': 'ElasticNet',
    'xg_boost': 'XGBRegressor', 
    'DecisionTreeRegressor': 'DecisionTreeRegressor',
    'SVM': 'SVR',
    'SGD': 'SGDRegressor',
    'KNN': 'KNeighborsRegressor',
    'extra_random_trees': 'ExtraTreesRegressor',
    'neural_network': 'MLPRegressor',
    }
        self.D_Classifier = {
    'RandomForestClassifier': 'RandomForestClassifier',
    'GBTClassifier': 'GradientBoostingClassifier',
    'LogisticRegression': 'LogisticRegression',
    'xg_boost': 'XGBClassifier',  
    'DecisionTreeClassifier': 'DecisionTreeClassifier',
    'SVM': 'SVC',
    'SGD': 'SGDClassifier',
    'KNN': 'KNeighborsClassifier',
    'extra_random_trees': 'ExtraTreesClassifier',
    'neural_network': 'MLPClassifier',
}
        
    def load_json(self, json_path):
        try: 
            with open(json_path, "r") as file: 
                rtf_content = file.read()
                plain_text = rtf_to_text(rtf_content) 
            return json.loads(plain_text)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return None
    def load_dataset(self, dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            print('Data read successfully')
        except FileNotFoundError as e:
            print(e)
            try:
                path = input('Please provide the path of the dataset:')
                df = pd.read_csv(path)
                print('Data read successfully')
            except Exception as e:
                print(e)
                return None
        return df
    
    def parse_json(self, json_data):
        target = json_data['design_state_data']['target']
        train = json_data['design_state_data']['train']
        feature_handling = json_data['design_state_data']['feature_handling']
        feature_generation = json_data['design_state_data']['feature_generation']
        feature_reduction = json_data['design_state_data']['feature_reduction']
        algorithms = json_data['design_state_data']['algorithms']
        return target, train, feature_handling, feature_generation, feature_reduction, algorithms
    
    def create_transformer(self):
        imputer = []
        for feature, details in self.feature_handling.items():
            if details['is_selected']:
                self.col_n.append(feature)
                if details['feature_variable_type'] == 'numerical' and details['feature_details']['missing_values'] == 'Impute':
                    strategy = 'mean' if details['feature_details']['impute_with'] == 'Average of values' else 'constant'
                    if strategy == 'mean':
                        imputer.append((feature, SimpleImputer(strategy=strategy), [self.col_n.index(feature)]))
                    elif strategy == 'constant':
                        fill_value = details['feature_details'].get('impute_value', 0)
                        imputer.append((feature, SimpleImputer(strategy=strategy, fill_value=fill_value), [self.col_n.index(feature)]))
                if details['feature_variable_type'] == 'text' and details['feature_details']["text_handling"] == "Tokenize and hash":
                    imputer.append((feature, OrdinalEncoder(), [self.col_n.index(feature)]))
        return ColumnTransformer(imputer, remainder='passthrough', n_jobs=-1)
    def pipe1(self):
        pipe=Pipeline(steps=[('preprocessor', self.trs),self.feature_generation,])
        return pipe.fit_transform(self.df)
    
    def apply_feature_reduction(self, feature_reduction):
        method = feature_reduction['feature_reduction_method']
        if method == 'Corr with Target':
            return ('feature_selection', SelectKBest(score_func=f_regression, k=int(feature_reduction['num_of_features_to_keep'])))
        elif method == 'Tree-based':
            if self.target['prediction_type'] == 'Regression':
                return ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=int(feature_reduction['num_of_trees']),
                                                                        max_depth=int(feature_reduction['depth_of_trees'])),
                                                                        max_features=int(feature_reduction['num_of_features_to_keep']),
                                                                        threshold = -np.inf))
            elif self.target['prediction_type'] == 'Classifier':
                return ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=int(feature_reduction['num_of_trees']),
                                                                        max_depth=int(feature_reduction['depth_of_trees'])),
                                                                        max_features=int(feature_reduction['num_of_features_to_keep']),
                                                                        threshold=-np.inf))
        elif method == 'PCA':
            return ('feature_selection', PCA(n_components=int(feature_reduction['num_of_features_to_keep'])))

        
    def train_test(self, X, y):
        config = self.train
        policy = config.get("policy", "Split the dataset")
        sampling_method = config.get("sampling_method", "No sampling(whole data)")
        split = config.get("split", "Randomly")
        k_fold = config.get("k_fold", False)
        train_ratio = config.get("train_ratio", 0.8)
        random_seed = config.get("random_seed", 0)
        
        if policy != "Split the dataset" or sampling_method != "No sampling(whole data)":
            raise ValueError("Only 'Split the dataset' and 'No sampling(whole data)' policies are supported.")
        
        if k_fold:
            kf = KFold(n_splits=5, shuffle=(split == "Randomly"), random_state=random_seed)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-train_ratio, random_state=random_seed, shuffle=(split == "Randomly")
            )
            return X_train, X_test, y_train, y_test

    def generate_param_grid(self):
        param_grid = {}
        for algo_name, algo_details in self.algorithms.items():
            if algo_details["is_selected"]:
                grid = {}

                # Random Forest parameters
                if algo_name in ["RandomForestClassifier", "RandomForestRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor"]:
                    if "min_trees" in algo_details and "max_trees" in algo_details:
                        grid["n_estimators"] = list(range(algo_details["min_trees"], algo_details["max_trees"] + 1))
                    if "min_depth" in algo_details and "max_depth" in algo_details:
                        grid["max_depth"] = list(range(algo_details["min_depth"], algo_details["max_depth"] + 1))
                    if "min_samples_per_leaf_min_value" in algo_details and "min_samples_per_leaf_max_value" in algo_details:
                        grid["min_samples_leaf"] = list(range(algo_details["min_samples_per_leaf_min_value"], algo_details["min_samples_per_leaf_max_value"] + 1))
                    if "min_samples_per_leaf" in algo_details:
                        grid["min_samples_leaf"] = algo_details["min_samples_per_leaf"]

                # Gradient Boosted Trees parameters
                if algo_name in ["GBTClassifier", "GBTRegressor"]:
                    if "num_of_BoostingStages" in algo_details:
                        grid["n_estimators"] = algo_details["num_of_BoostingStages"]
                    if "min_stepsize" in algo_details and "max_stepsize" in algo_details:
                        grid["learning_rate"] = [x / 100.0 for x in range(int(algo_details["min_stepsize"] * 100), int(algo_details["max_stepsize"] * 100) + 1)]
                    if "min_depth" in algo_details and "max_depth" in algo_details:
                        grid["max_depth"] = list(range(algo_details["min_depth"], algo_details["max_depth"] + 1))
                                                                               

                # Linear and Logistic Regression parameters
                if algo_name in ["LinearRegression", "LogisticRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression"]:
                    if "min_iter" in algo_details and "max_iter" in algo_details:
                        grid["max_iter"] = list(range(algo_details["min_iter"], algo_details["max_iter"] + 1))
                    if "min_regparam" in algo_details and "max_regparam" in algo_details:
                        grid["alpha"] = [x / 100.0 for x in range(int(algo_details["min_regparam"] * 100), int(algo_details["max_regparam"] * 100) + 1)]
                    if "min_elasticnet" in algo_details and "max_elasticnet" in algo_details:
                        grid["l1_ratio"] = [x / 100.0 for x in range(int(algo_details["min_elasticnet"] * 100), int(algo_details["max_elasticnet"] * 100) + 1)]

                # XGBoost parameters
                if algo_name == "xg_boost":
                    if "max_depth_of_tree" in algo_details:
                        grid["max_depth"] = algo_details["max_depth_of_tree"]
                    if "learningRate" in algo_details:
                        grid["learning_rate"] = algo_details["learningRate"]
                    if "l1_regularization" in algo_details:
                        grid["reg_alpha"] = algo_details["l1_regularization"]
                    if "l2_regularization" in algo_details:
                        grid["reg_lambda"] = algo_details["l2_regularization"]
                    if "gamma" in algo_details:
                        grid["gamma"] = algo_details["gamma"]
                    if "min_child_weight" in algo_details:
                        grid["min_child_weight"] = algo_details["min_child_weight"]
                    if "sub_sample" in algo_details:
                        grid["subsample"] = algo_details["sub_sample"]
                    if "col_sample_by_tree" in algo_details:
                        grid["colsample_bytree"] = algo_details["col_sample_by_tree"]

                # SVM parameters
                if algo_name == "SVM":
                    if "c_value" in algo_details:
                        grid["C"] = algo_details["c_value"]
                    if "custom_gamma_values" in algo_details and algo_details["custom_gamma_values"]:
                        grid["gamma"] = ["scale", "auto"]

                # KNN parameters
                if algo_name == "KNN":
                    if "k_value" in algo_details:
                        grid["n_neighbors"] = algo_details["k_value"]
                    if "p_value" in algo_details:
                        grid["p"] = [algo_details["p_value"]]

                # Extra Random Trees parameters
                if algo_name == "extra_random_trees":
                    if "num_of_trees" in algo_details:
                        grid["n_estimators"] = algo_details["num_of_trees"]
                    if "max_depth" in algo_details:
                        grid["max_depth"] = algo_details["max_depth"]
                    if "min_samples_per_leaf" in algo_details:
                        grid["min_samples_leaf"] = algo_details["min_samples_per_leaf"]

                # Neural Network parameters
                if algo_name == "neural_network":
                    if "hidden_layer_sizes" in algo_details:
                        grid["hidden_layer_sizes"] = algo_details["hidden_layer_sizes"]
                    if "activation" in algo_details and algo_details["activation"]:
                        grid["activation"] = [algo_details["activation"]]
                    if "solver" in algo_details:
                        grid["solver"] = [algo_details["solver"]]
                    if "max_iterations" in algo_details and algo_details["max_iterations"] > 0:
                        grid["max_iter"] = [algo_details["max_iterations"]]

                if grid:
                    if self.target['prediction_type'].lower()=='regression':
                        if algo_name in self.D_Regressor.keys():
                            grid={ i:grid[i] for i in set(grid.keys()).intersection(set(eval(f'{self.D_Regressor[algo_name]}()').get_params()))}
                            param_grid[algo_name] = {f'{algo_name}__{key}': value for key, value in grid.items()}
                    if self.target['prediction_type'].lower()=='classification':
                        if algo_name in self.D_Classifier.keys():
                            grid={ i:grid[i] for i in set(grid.keys()).intersection(set(eval(f'{self.D_Regressor[algo_name]}()').get_params()))}
                            param_grid[algo_name] = {f'{algo_name}__{key}': value for key, value in grid.items()}
        return param_grid
    
    def get_selected_models(self):
        selected_models = []
        prediction_type = self.target['prediction_type']
        for algo_name, algo_details in self.algorithms.items():
            if algo_details['is_selected']:
                if prediction_type.lower() == 'regression':
                    if algo_name in self.D_Regressor.keys():
                        selected_models.append(Pipeline(steps=[
                            self.feature_red,
                            (f'{algo_name}', eval(f"{self.D_Regressor[algo_name]}()"))
                        ]))
                elif prediction_type.lower() == 'classifier':
                    if algo_name in self.D_Classifier.keys():
                        selected_models.append(Pipeline(steps=[
                            self.feature_red,
                            (f'{algo_name}', eval(f'{self.D_Classifier[algo_name]}()'))
                        ]))
        return selected_models
    
    def compute_general_metrics(self, y_true, y_pred, task_type, compute_lift_at=0, cost_matrix=None, optimize_model_hyperparameters_for="AUC", optimize_threshold_for="F1 Score"):
        metrics_dict = {}
        
        if cost_matrix is None:
            cost_matrix = {
                "gain_tp": 1,
                "gain_fp": 0,
                "gain_fn": 0,
                "gain_tn": 0
            }
        
        if task_type.lower == "classification":
            if len(np.unique(y_pred)) > 2 and np.min(y_pred) >= 0 and np.max(y_pred) <= 1:
                y_pred_labels = (y_pred >= 0.5).astype(int)
            else:
                y_pred_labels = y_pred
            
            if optimize_model_hyperparameters_for == "AUC":
                metrics_dict["AUC"] = roc_auc_score(y_true, y_pred)
            
            if optimize_threshold_for == "F1 Score":
                metrics_dict["F1 Score"] = f1_score(y_true, y_pred_labels)
            
            gain = 0
            for yt, yp in zip(y_true, y_pred_labels):
                if yt == 1 and yp == 1:
                    gain += cost_matrix["gain_tp"]
                elif yt == 0 and yp == 1:
                    gain += cost_matrix["gain_fp"]
                elif yt == 1 and yp == 0:
                    gain += cost_matrix["gain_fn"]
                elif yt == 0 and yp == 0:
                    gain += cost_matrix["gain_tn"]
            metrics_dict["Custom Gain"] = gain

            if compute_lift_at > 0:
                sorted_predictions = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
                top_n = int(len(sorted_predictions) * compute_lift_at)
                y_true_top_n = [yt for yt, _ in sorted_predictions[:top_n]]
                lift_gain = sum(y_true_top_n) * cost_matrix["gain_tp"]
                metrics_dict["Lift at {:.0%}".format(compute_lift_at)] = lift_gain

        elif task_type.lower() == "regression":
            metrics_dict["MSE"] = mean_squared_error(y_true, y_pred)
            metrics_dict["MAE"] = mean_absolute_error(y_true, y_pred)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                metrics_dict["R2"] = r2_score(y_true, y_pred)
        else:
            raise ValueError("Invalid task_type. Must be 'classification' or 'regression'.")
        
        return metrics_dict
    
    def custom_grid_search_cv(self, estimator_list, param_grid, X, y, task_type):
        tscv = TimeSeriesSplit(n_splits=6)
        scoring = 'r2' if self.target['prediction_type'].lower() == 'regression' else 'accuracy'
        best_estimators = []
        
        for estimator, algo_name in zip(estimator_list, param_grid):
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid[algo_name],
                scoring=scoring,
                cv=tscv,
                n_jobs=5,
                verbose=1
            )
            grid_search.fit(X, y.values.ravel())
            metrics_dict = self.compute_general_metrics(y, grid_search.predict(X), task_type)
            print(algo_name)
            print(metrics_dict)
            best_estimators.append((grid_search.best_estimator_))
        
        return best_estimators
    
    def run(self):
        X_train, X_test, y_train, y_test = self.train_test(self.df_.drop(self.target['target'], axis=1), self.df_[[self.target['target']]])
        param_grid = self.generate_param_grid()
        selected_models = self.get_selected_models()
        best_models = self.custom_grid_search_cv(selected_models, param_grid, X_train, y_train, self.target['prediction_type'])
        return best_models

# Example usage:
# model = MLModel(r'E:\New folder\algoparams_from_ui.json.rtf')
# best_models = model.run()
# print(best_models)
