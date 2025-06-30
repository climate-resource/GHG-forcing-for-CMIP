"""
Main workflow
"""

from prefect import flow

from ghg_forcing_for_cmip_comparison.get_datasets import get_data_flow


@flow(
    name="run_pipeline_comparison",
    description="Run main pipeline for comparison of EO vs. CMIP",
)
def run_pipeline_comparison():
    """
    Run main workflow for GHG-forcing-for-CMIP-comparison
    """
    get_data_flow()


if __name__ == "__main__":
    run_pipeline_comparison()
