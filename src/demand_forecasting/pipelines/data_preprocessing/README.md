# Data Preprocessing pipeline
## Overview

This modular pipeline preprocesses the raw data (`preprocess_data_node`) and defines data that will be in all executions (`defines_global_data_node`).

## Pipeline inputs

### `raw_data`

|      |                    |
| ---- | ------------------ |
| Type | `polars.DataFrame` |
| Description | Raw temporal data |


## Pipeline outputs

### `processed_data`

|      |                    |
| ---- | ------------------ |
| Type | `polars.DataFrame` |
| Description | Preprocessed temporal data |

### `unique_ids`

|      |                    |
| ---- | ------------------ |
| Type | `List[str]` |
| Description | List containing the data unique ids (store_item) |

### `global_data`

|      |                    |
| ---- | ------------------ |
| Type | `GlobalData` |
| Description | GlobalData class object |
