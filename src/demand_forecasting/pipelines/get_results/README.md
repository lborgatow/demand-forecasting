# Get Results pipeline
## Overview

his pipeline obtains the results of each model's predictions for each combination using parallel execution (`run_parallel_predictions_node`).

## Pipeline inputs

### `global_data`

|      |                    |
| ---- | ------------------ |
| Type | `GlobalData` |
| Description | GlobalData class object |

### `unique_ids`

|      |                    |
| ---- | ------------------ |
| Type | `List[str]` |
| Description | List containing the data unique ids |


## Pipeline outputs

### `models_results`

|      |                    |
| ---- | ------------------ |
| Type | `polars.LazyFrame` |
| Description | DataFrame with the results of all models for each combination |


