from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, define_global_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs="raw_data",
                outputs=["processed_data", "unique_ids"],
                name="preprocess_data_node",
            ),
            node(
                func=define_global_data,
                inputs=["processed_data", "parameters"],
                outputs="global_data",
                name="define_global_data_node",
            ),
        ],
        namespace="pipe_data_preprocessing",
        inputs="raw_data",
        outputs=["processed_data", "unique_ids", "global_data"],
    )