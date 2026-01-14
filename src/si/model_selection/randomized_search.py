
import numpy as np
import itertools
from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model: Model, dataset: Dataset, hyperparameter_grid: dict, 
                         cv: int = 5, n_iter: int = 10, scoring: callable = None) -> dict:
    """
    Performs a randomized search cross validation on a model.

    Parameters
    ----------
    model : Model
        The model to cross validate.
    dataset : Dataset
        The dataset to cross validate on.
    hyperparameter_grid : dict
        The hyperparameter grid to use.
    cv : int
        The cross validation folds.
    n_iter : int
        Number of hyperparameter random combinations to test.
    scoring : callable
        The scoring function to use.

    Returns
    -------
    results : dict
        The results of the randomized search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # Validate the parameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {type(model).__name__} does not have parameter '{parameter}'.")

    # Generate all possible combinations
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))

    # Select n_iter random combinations
    num_combinations = len(all_combinations)
    n_selection = min(n_iter, num_combinations)
    
    # Choose unique random indices
    random_indices = np.random.choice(num_combinations, size=n_selection, replace=False)
    
    # Filter only the selected combinations
    selected_combinations = [all_combinations[i] for i in random_indices]

    results = {'scores': [], 'hyperparameters': []}

    # Iterate through selected combinations
    for combination in selected_combinations:
        # Reconstruct the dictionary for the current combination
        parameters = {param: value for param, value in zip(hyperparameter_grid.keys(), combination)}

        # Set parameters in the model
        for param, value in parameters.items():
            setattr(model, param, value)

        # Cross validate
        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Store results
        results['scores'].append(np.mean(scores))
        results['hyperparameters'].append(parameters)

    # Find best result
    best_idx = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score'] = results['scores'][best_idx]
    
    return results