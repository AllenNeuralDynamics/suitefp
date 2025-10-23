import pandas as pd
from suitefp.qc import FiberPhotometryData, FiberPhotometryQCEvaluator


def run_specific_raw_metrics(fp: str, fiber_channel: str, metric_names: list):
    """Example function to run specific raw metrics on a color and fiber channel."""
    data = FiberPhotometryData.from_csv(fp, fiber_channel=fiber_channel)
    evaluator = FiberPhotometryQCEvaluator(raw_metrics=metric_names)
    raw_results = evaluator.evaluate_raw_metrics(data)
    return raw_results


def run_raw_processed_metrics(
    fp: str, fiber_channel: str, raw_metrics: list, processed_metrics: list
):
    """Example function to run raw and processed metrics on a color and fiber channel."""
    data = FiberPhotometryData.from_csv(fp, fiber_channel=fiber_channel)
    evaluator = FiberPhotometryQCEvaluator(
        raw_metrics=raw_metrics, processed_metrics=processed_metrics
    )
    raw_results = evaluator.evaluate_raw_metrics(data)
    processed_results = evaluator.evaluate_processed_metrics(data)
    return raw_results, processed_results


def run_all_metrics(fp: str, fiber_channel: str):
    """Example function to run all QC metrics on a given file and fiber channel."""
    data = FiberPhotometryData.from_csv(fp, fiber_channel=fiber_channel)
    evaluator = FiberPhotometryQCEvaluator()
    all_results = evaluator.evaluate_all_metrics(data)
    return all_results


if __name__ == "__main__":
    green_fp = "green.csv"
    red_fp = "red.csv"
    iso_fp = "iso.csv"

    # Run all metrics on each file and one channel
    green_results = run_all_metrics(green_fp, "Fiber_0")
    red_results = run_all_metrics(red_fp, "Fiber_0")
    iso_results = run_all_metrics(iso_fp, "Fiber_0")

    # Run raw and processed metrics on red file, Fiber_2
    red_raw_results, red_processed_results = run_raw_processed_metrics(
        red_fp,
        "Fiber_2",
        raw_metrics=["NoFiberNaNMetric"],
        processed_metrics=["ZScoreMetric"],
    )

    # Run specific raw metrics on green file, Fiber_1
    specific_green_results = run_specific_raw_metrics(
        green_fp,
        "Fiber_1",
        metric_names=["DataDurationMetric", "NoFiberNaNMetric"],
    )
