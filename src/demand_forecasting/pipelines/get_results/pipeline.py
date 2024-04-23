from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_parallel_predictions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_parallel_predictions,
                inputs=["global_data", "unique_ids", "parameters"],
                outputs=None,
                name="preprocess_data_node",
            ),
        ],
        namespace="pipe_get_results",
        inputs=["global_data", "unique_ids"],
        outputs=None,
    )