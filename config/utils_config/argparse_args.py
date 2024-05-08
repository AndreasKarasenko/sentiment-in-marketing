arguments = [
    {
        "arg": "--config_dir",
        "type": str,
        "default": "./config/model_config/search_space.json",
        "help": "Path to the search space config file",
    },
    {
        "arg": "--data_dir",
        "type": str,
        "default": "./samples/",
        "help": "Path to the data directory",
    },
    {
        "arg": "--model_dir",
        "type": str,
        "default": "./models/",
        "help": "Path to the model directory",
    },
    {
        "arg": "--optimized_dir",
        "type": str,
        "default": "./config/optimized/",
        "help": "Path to the optimized directory",
    },
    {
        "arg": "--input_var",
        "type": str,
        "default": "content",
        "help": "Name of the input variable",
    },
    {
        "arg": "--target_var",
        "type": str,
        "default": "label",
        "help": "Name of the target variable",
    },
    {"arg": "--verbose", "type": int, "default": 1, "help": "Verbosity level"},
    {
        "arg": "--njobs",
        "type": int,
        "default": 1,
        "help": "Number of jobs for gridsearch",
    },
    {
        "arg": "--data_config",
        "type": str,
        "default": "./config/data_config/",
        "help": "Input and output data and dataset names config directory",
    },
]
