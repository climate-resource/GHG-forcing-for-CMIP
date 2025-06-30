"""
Main workflow
"""

from prefect import flow

from ghg_forcing_for_cmip_comparison.bin_dataset_gb import bin_dataset_flow
from ghg_forcing_for_cmip_comparison.get_datasets import get_data_flow


@flow(
    name="run_pipeline_comparison",
    description="Run main pipeline for comparison of EO vs. CMIP",
)
def run_pipeline_comparison(
    gas: str, quantile: float, save_to_path: str = "data/downloads"
) -> None:
    """
    Run main workflow for GHG-forcing-for-CMIP-comparison
    """
    get_data_flow(save_to_path=save_to_path)

    bin_dataset_flow(path_to_csv=save_to_path, gas=gas, quantile=quantile)


if __name__ == "__main__":
    run_pipeline_comparison()
