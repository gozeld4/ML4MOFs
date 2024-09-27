## Python 3.9, 

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import os

def train_test_plot(y_train, y_train_pred, y_test, y_test_pred, write_dir):
    plt.rcParams['axes.linewidth'] = 1.5
    fig_train = plt.figure(figsize=[6.4, 6.4])
    ax = fig_train.gca()
    corr_coeff, p_value = pearsonr(y_train, y_train_pred)
    ax_min = min(min(y_train), min(y_train_pred))
    ax_max = max(max(y_train), max(y_train_pred))
    ax_min = int(ax_min)
    ax_max = int(ax_max)
    parity = np.linspace(ax_min, ax_max)
    ax.scatter(y_train, y_train_pred, marker='o', color=(0, 0, 1, 0.5))
    ax.set_xlabel('Actual water uptake (cm$\mathbf{^3}$/cm$\mathbf{^3}$ framework)', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_ylabel('Predicted water uptake (cm$\mathbf{^3}$/cm$\mathbf{^3}$ framework)', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_title(f'Training: R = {format(corr_coeff, ".2")}', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.plot(parity, parity, color=(1, 0, 0, 0.5), linewidth=1.5)
    plt.xticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.yticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.tick_params(axis='both', direction='in', width=1, length=6)
    #plt.show()
    fig_train.savefig(f'{write_dir}train.pdf', dpi=600, bbox_inches='tight')

    fig_test = plt.figure(figsize=[6.4, 6.4])
    ax = fig_test.gca()
    corr_coeff, p_value = pearsonr(y_test, y_test_pred)
    ax_min = min(min(y_test), min(y_test_pred))
    ax_max = max(max(y_test), max(y_test_pred))
    ax_min = int(ax_min)
    ax_max = int(ax_max)
    parity = np.linspace(ax_min, ax_max)
    ax.scatter(y_test, y_test_pred, marker='o', color=(0, 0, 1, 0.5))
    ax.set_xlabel('Actual water uptake (cm$\mathbf{^3}$/cm$\mathbf{^3}$ framework)', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_ylabel('Predicted water uptake (cm$\mathbf{^3}$/cm$\mathbf{^3}$ framework)', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_title(f'Test: R = {format(corr_coeff, ".2")}', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.plot(parity, parity, color=(1, 0, 0, 0.5), linewidth=2)
    plt.xticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.yticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.tick_params(axis='both', direction='in', width=1, length=6)
    #plt.show()
    fig_test.savefig(f'{write_dir}test.pdf', dpi=600, bbox_inches='tight')

## Manual RFA
def recursive_feature_addition(X, y, model, scoring='neg_mean_squared_error', cv=3):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    
    ## Identify the top 5 features
    model.fit(X, y)

    importances = model.feature_importances_

    top_n = 5
    indices = np.argsort(importances)[::-1][:top_n]

    
    for index in indices:
        selected_features.append(index)
        remaining_features.remove(index)
    
    best_score = cross_val_score(model, X[:, selected_features], y, cv=cv, scoring=scoring).mean()
    print(best_score)

    while remaining_features:
        scores = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_subset = X[:, features_to_test]
            score = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring).mean()
            scores.append((score, feature))
        
        scores.sort(reverse=True)
        best_new_score, best_new_feature = scores[0]
        
        if (best_new_score - best_score)/abs(best_score) > 0.01:
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            best_score = best_new_score
            print(f"Selected feature {best_new_feature} with score {best_new_score}")
        else:
            break
            
    return selected_features


def recursive_feature_addition_krr(X, y, model, scoring='neg_mean_squared_error', cv=3):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    
    ## Identify the top 5 features
    dummy_model = RandomForestRegressor(random_state=r_state)
    dummy_model.fit(X, y)

    importances = dummy_model.feature_importances_

    top_n = 5
    indices = np.argsort(importances)[::-1][:top_n]

    for index in indices:
        selected_features.append(index)
        remaining_features.remove(index)
    
    best_score = cross_val_score(model, X[:, selected_features], y, cv=cv, scoring=scoring).mean()
    print(best_score)

    while remaining_features:
        scores = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_subset = X[:, features_to_test]
            score = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring).mean()
            scores.append((score, feature))
        
        scores.sort(reverse=True)
        best_new_score, best_new_feature = scores[0]
        
        if (best_new_score - best_score)/abs(best_score) > 0.01:
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            best_score = best_new_score
            print(f"Selected feature {best_new_feature} with score {best_new_score}")
        else:
            break
            
    return selected_features


RandomState = [42, 84, 168, 336]
data_splits = ['1inorg_1edge', '1inorg_1org_1edge', '2inorg_1edge', 'all']
write_csv_name = 'RACs_zeopp_model_performance_no_mlp.csv'

write_error_dict = {'dataset': [], 'model': [], 'random_state': [], 'train_MAE': [], 'test_MAE': [], 'train R2': [], 'test R2': []}

for r_state in RandomState:
    for split in data_splits:

        save_dir = f'./ML_model_ver4/random_state_{r_state}/ML_models_{split}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = pd.read_csv(f'combined_data_frame_{split}.csv')
        column_names = data.columns


        column_to_drop = ['data_type', 'net', 'KVRH', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 'D_func-S-0-all', 
                  'D_func-T-0-all', 'D_func-Z-0-all', 'D_func-alpha-0-all', 'D_func-chi-0-all', 'D_lc-I-0-all', 
                  'D_lc-I-1-all', 'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-T-0-all', 'D_lc-Z-0-all', 
                  'D_lc-alpha-0-all', 'D_lc-chi-0-all', 'lc-I-0-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all', 
                  'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-T-0-all', 'D_mc-Z-0-all', 'D_mc-chi-0-all', 'mc-I-0-all']
        
        X = data.drop(column_to_drop, axis=1)
        print(X.shape)
        y = data['KVRH']

        
        indices = np.arange(len(X.drop('name', axis=1)))

        X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X.drop('name', axis=1), y, indices, test_size=0.2, random_state=r_state, shuffle=True)

        train_feature_order = X.drop('name', axis=1).columns
        print(len(train_feature_order))
        print(list(train_feature_order))
        scaler = StandardScaler().fit(X_train)

        X_train_original = scaler.transform(X_train)
        X_test_original = scaler.transform(X_test)

        ## saving the scaler
        with open(f'{save_dir}scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)


        ## Random forest training
        rf_regressor = RandomForestRegressor(random_state=r_state)

        X_train_scaled = X_train_original
        X_test_scaled = X_test_original

        selected_features_indices = recursive_feature_addition(X_train_scaled, y_train, rf_regressor)

        X_train_selected = X_train_scaled[:, selected_features_indices]
        X_test_selected = X_test_scaled[:, selected_features_indices]

        features_selected = []

        for feature_index in selected_features_indices:
            features_selected.append(X.drop('name', axis=1).columns[feature_index])

        write_dir = f'{save_dir}/random_forest/'
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        with open(f'{write_dir}selected_features_1%.txt', 'w') as write_file:
            write_file.write(str(features_selected))


        rf_regressor = RandomForestRegressor(random_state=r_state)

        # Define the hyperparameter grid to search
        param_grid = {
            'n_estimators': [80, 160, 320, 640, 1280], 
            'min_samples_split': [6, 8, 10, 12, 14], 
            'max_depth': [2, 4, 8, 12, 16]
        }


        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=3)


        # Fit the model to the training data
        grid_search.fit(X_train_selected, y_train)

        # Print the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        with open(f'{write_dir}best_hyperparameters.txt', 'w') as write_file:
            write_file.write(str(best_params))

        # Get the best model from the grid search
        best_rf_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_train_pred = best_rf_model.predict(X_train_selected)
        y_test_pred = best_rf_model.predict(X_test_selected)

        # Evaluate the mean squared error
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        print(f"Mean Squared Error on Train Set: {mse_train}")
        print(f"Mean Squared Error on Test Set: {mse_test}")

        write_error_dict['dataset'].append(split)
        write_error_dict['random_state'].append(r_state)
        write_error_dict['model'].append('random forest')
        write_error_dict['train_MAE'].append(mse_train**0.5)
        write_error_dict['test_MAE'].append(mse_test**0.5)
        write_error_dict['train R2'].append(r2_train)
        write_error_dict['test R2'].append(r2_test)
        ## Saving the files
    
        with open(f'{write_dir}best_rf_model.pkl', 'wb') as file:
            pickle.dump(best_rf_model, file)


        pd.concat([pd.DataFrame(list(X['name'][index_train])), pd.DataFrame(list(y_train)), pd.DataFrame(y_train_pred.flatten())], axis=1).to_csv(f'{write_dir}train_results.csv', index=None, header=['MOF_name', 'Actual', 'Predicted'])
        pd.concat([pd.DataFrame(list(X['name'][index_test])), pd.DataFrame(list(y_test)), pd.DataFrame(y_test_pred.flatten())], axis=1).to_csv(f'{write_dir}test_results.csv', index=None,  header=['MOF_name', 'Actual', 'Predicted'])

        train_test_plot(y_train, y_train_pred, y_test, y_test_pred, write_dir)

        gb_regressor = GradientBoostingRegressor(random_state=r_state)

        X_train_scaled = X_train_original
        X_test_scaled = X_test_original

        selected_features_indices = recursive_feature_addition(X_train_scaled, y_train, gb_regressor)

        X_train_selected = X_train_scaled[:, selected_features_indices]
        X_test_selected = X_test_scaled[:, selected_features_indices]

        features_selected = []

        for feature_index in selected_features_indices:
            features_selected.append(X.drop('name', axis=1).columns[feature_index])

        write_dir = f'{save_dir}/gradient_boost/'
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)


        with open(f'{write_dir}selected_features_1%.txt', 'w') as write_file:
            write_file.write(str(features_selected))


        gbr = GradientBoostingRegressor(random_state=r_state)

        # Define the hyperparameter grid to search
        param_grid = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.4],
            'max_depth': [2, 3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4]
        }


        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=3)

        # Fit the model to the training data
        grid_search.fit(X_train_selected, y_train)

        # Print the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        with open(f'{write_dir}best_hyperparameters.txt', 'w') as write_file:
            write_file.write(str(best_params))

        # Get the best model from the grid search
        best_gb_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_train_pred = best_gb_model.predict(X_train_selected)
        y_test_pred = best_gb_model.predict(X_test_selected)

        # Evaluate the mean squared error
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        print(f"Mean Squared Error on Train Set: {mse_train}")
        print(f"Mean Squared Error on Test Set: {mse_test}")

        write_error_dict['dataset'].append(split)
        write_error_dict['random_state'].append(r_state)
        write_error_dict['model'].append('gradient boosting')
        write_error_dict['train_MAE'].append(mse_train**0.5)
        write_error_dict['test_MAE'].append(mse_test**0.5)
        write_error_dict['train R2'].append(r2_train)
        write_error_dict['test R2'].append(r2_test)

        ## Saving the files
        with open(f'{write_dir}best_gb_model.pkl', 'wb') as file:
            pickle.dump(best_gb_model, file)
            
        pd.concat([pd.DataFrame(list(X['name'][index_train])), pd.DataFrame(list(y_train)), pd.DataFrame(y_train_pred.flatten())], axis=1).to_csv(f'{write_dir}train_results.csv', index=None, header=['MOF_name', 'Actual', 'Predicted'])
        pd.concat([pd.DataFrame(list(X['name'][index_test])), pd.DataFrame(list(y_test)), pd.DataFrame(y_test_pred.flatten())], axis=1).to_csv(f'{write_dir}test_results.csv', index=None,  header=['MOF_name', 'Actual', 'Predicted'])
        train_test_plot(y_train, y_train_pred, y_test, y_test_pred, write_dir)


        krr = KernelRidge(kernel='laplacian')

        X_train_scaled = X_train_original
        X_test_scaled = X_test_original

        selected_features_indices = recursive_feature_addition_krr(X_train_scaled, y_train, krr)

        X_train_selected = X_train_scaled[:, selected_features_indices]
        X_test_selected = X_test_scaled[:, selected_features_indices]

        features_selected = []

        for feature_index in selected_features_indices:
            features_selected.append(X.drop('name', axis=1).columns[feature_index])

        write_dir = f'{save_dir}/krr_laplacian/'
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        with open(f'{write_dir}selected_features_1%.txt', 'w') as write_file:
            write_file.write(str(features_selected))


        krr = KernelRidge(kernel='laplacian')

        # Define the hyperparameter grid to search
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'gamma': [0.001, 0.01, 0.1, 1.0, 10.0],

        }
        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=krr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=3)

        # Fit the model to the training data
        grid_search.fit(X_train_selected, y_train)

        # Print the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        with open(f'{write_dir}best_hyperparameters.txt', 'w') as write_file:
            write_file.write(str(best_params))

        # Get the best model from the grid search
        best_krr_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_train_pred = best_krr_model.predict(X_train_selected)
        y_test_pred = best_krr_model.predict(X_test_selected)

        # Evaluate the mean squared error
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        print(f"Mean Squared Error on Train Set: {mse_train}")
        print(f"Mean Squared Error on Test Set: {mse_test}")

        write_error_dict['dataset'].append(split)
        write_error_dict['random_state'].append(r_state)
        write_error_dict['model'].append('KRR laplacian')
        write_error_dict['train_MAE'].append(mse_train**0.5)
        write_error_dict['test_MAE'].append(mse_test**0.5)
        write_error_dict['train R2'].append(r2_train)
        write_error_dict['test R2'].append(r2_test)

        with open(f'{write_dir}best_krr_model.pkl', 'wb') as file:
            pickle.dump(best_krr_model, file)
            
        pd.concat([pd.DataFrame(list(X['name'][index_train])), pd.DataFrame(list(y_train)), pd.DataFrame(y_train_pred.flatten())], axis=1).to_csv(f'{write_dir}train_results.csv', index=None, header=['MOF_name', 'Actual', 'Predicted'])
        pd.concat([pd.DataFrame(list(X['name'][index_test])), pd.DataFrame(list(y_test)), pd.DataFrame(y_test_pred.flatten())], axis=1).to_csv(f'{write_dir}test_results.csv', index=None,  header=['MOF_name', 'Actual', 'Predicted'])
        train_test_plot(y_train, y_train_pred, y_test, y_test_pred, write_dir)

        krr = KernelRidge(kernel='rbf')

        X_train_scaled = X_train_original
        X_test_scaled = X_test_original

        selected_features_indices = recursive_feature_addition_krr(X_train_scaled, y_train, krr)

        X_train_selected = X_train_scaled[:, selected_features_indices]
        X_test_selected = X_test_scaled[:, selected_features_indices]

        features_selected = []

        for feature_index in selected_features_indices:
            features_selected.append(X.drop('name', axis=1).columns[feature_index])

        write_dir = f'{save_dir}/krr_rbf/'
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        with open(f'{write_dir}selected_features_1%.txt', 'w') as write_file:
            write_file.write(str(features_selected))

        krr = KernelRidge(kernel='rbf')

        # Define the hyperparameter grid to search
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'gamma': [0.001, 0.01, 0.1, 1.0, 10.0],

        }
        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=krr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=3)

        # Fit the model to the training data
        grid_search.fit(X_train_selected, y_train)

        # Print the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        with open(f'{write_dir}best_hyperparameters.txt', 'w') as write_file:
            write_file.write(str(best_params))

        # Get the best model from the grid search
        best_krr_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_train_pred = best_krr_model.predict(X_train_selected)
        y_test_pred = best_krr_model.predict(X_test_selected)

        # Evaluate the mean squared error
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        print(f"Mean Squared Error on Train Set: {mse_train}")
        print(f"Mean Squared Error on Test Set: {mse_test}")


        write_error_dict['dataset'].append(split)
        write_error_dict['random_state'].append(r_state)
        write_error_dict['model'].append('KRR RBF')
        write_error_dict['train_MAE'].append(mse_train**0.5)
        write_error_dict['test_MAE'].append(mse_test**0.5)
        write_error_dict['train R2'].append(r2_train)
        write_error_dict['test R2'].append(r2_test)

        with open(f'{write_dir}best_krr_model.pkl', 'wb') as file:
            pickle.dump(best_krr_model, file)
            
        pd.concat([pd.DataFrame(list(X['name'][index_train])), pd.DataFrame(list(y_train)), pd.DataFrame(y_train_pred.flatten())], axis=1).to_csv(f'{write_dir}train_results.csv', index=None, header=['MOF_name', 'Actual', 'Predicted'])
        pd.concat([pd.DataFrame(list(X['name'][index_test])), pd.DataFrame(list(y_test)), pd.DataFrame(y_test_pred.flatten())], axis=1).to_csv(f'{write_dir}test_results.csv', index=None,  header=['MOF_name', 'Actual', 'Predicted'])
        train_test_plot(y_train, y_train_pred, y_test, y_test_pred, write_dir)


pd.DataFrame.from_dict(write_error_dict).to_csv(write_csv_name, index=None)


     






