"""
Main workflow
"""

from prefect import flow

from ghg_forcing_for_cmip.data_comparison.bin_dataset_gb import bin_dataset_flow
from ghg_forcing_for_cmip.data_comparison.combine_datasets import join_datasets_flow
from ghg_forcing_for_cmip.data_comparison.get_datasets import get_data_flow
from ghg_forcing_for_cmip.data_comparison.interpolate_dataset_gb import (
    interpolation_flow,
)
from ghg_forcing_for_cmip.data_comparison.vertical_to_dataset_gb import (
    add_vertical_flow,
)


@flow(
    name="run_pipeline_comparison",
    description="Run main pipeline for comparison of EO vs. CMIP",
)
def run_pipeline_comparison(
    gas: str, quantile: float, save_to_path: str = "data/downloads"
) -> None:
    """
    Run main workflow for ghg-forcing-for-cmip
    """
    get_data_flow(save_to_path=save_to_path)

    bin_dataset_flow(path_to_csv=save_to_path, gas=gas, quantile=quantile)

    interpolation_flow(path_to_csv=save_to_path, gas=gas)

    add_vertical_flow(path_to_csv=save_to_path, gas=gas)

    join_datasets_flow(path_to_csv=save_to_path, gas=gas)


if __name__ == "__main__":
    run_pipeline_comparison(gas="ch4", quantile=0.5)  # type: ignore
