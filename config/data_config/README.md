For the data config we define input and output configs
Both are saved as JSON and should have the following structures.

Since I'm mostly working with csv files they usually specify pandas DataFrame columns

input_config.json
```json
{
    "input_var": "text"
}
```
target_config.json
```json
{
    "target_vars": ["ATT", "BI", "PE", "PEOU", "PI", "PU"]
}
```