### function to save results (metrics, runtime, best parameters) from run_all.py or run.py to results/train/FILENAME
### where FILENAME is the name of the model, the construct that was learned, and the date and time
### the results are saved in a json file
### also saves the gridsearch object as a pickle file
### TODO adjust for cuML
import joblib

def save_results(filename, model_name, dataset, metrics, args, grid = None):
    """Save results (metrics, runtime, best parameters) from run_all.py or run.py to results/train/FILENAME
    where FILENAME is the name of the model, the construct that was learned, and the date and time
    the results are saved in a json file

    Args:
        filename: name of the file to save the results
        model_name: name of the model
        construct: construct that was learned
        metrics: evaluation metrics
        args: arguments from argparse
        grid: optional gridsearchcv results object
    """
    import os
    import json
    from datetime import datetime

    ### create a directory to save the results
    if not os.path.exists("results/train/" + model_name):
        os.makedirs("results/train/" + model_name)

    ### save the results
    with open("results/train/" + model_name + "/" + filename + ".json",
        "w",
    ) as f:
        if grid:
            json.dump(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "metrics": metrics,
                    "best_hyperparameters": grid.best_params_,
                    "arguments": vars(args),
                    "search_space": json.load(open(args.config_dir, "r"))[model_name]["hyperparameters"],
                    # "grid_search_results": grid.cv_results_,
                    "best_score": grid.best_score_,
                    "best_estimator": str(grid.best_estimator_),
                    "best_params": grid.best_params_,
                    "scorer": str(grid.scorer_),
                    "refit_time": grid.refit_time_,
                    "mean_fit_time": list(grid.cv_results_["mean_fit_time"]),
                    "multimetric": grid.multimetric_,
                    "best_index": int(grid.best_index_),
                    "cv": int(grid.cv),
                    "n_splits": grid.n_splits_,
                },
                f,
                indent=4,
            )
            grid_pkl_path = "results/train/" + model_name + "/" + filename + ".pkl"
            joblib.dump(grid, grid_pkl_path)
        else:
            json.dump(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "metrics": metrics,
                    "arguments": vars(args),
                },
                f,
                indent=4,
            )