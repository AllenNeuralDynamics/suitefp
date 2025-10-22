import pandas as pd
from photo_qc import FiberPhotometryData, FiberPhotometryQCEvaluator

# Test the new simplified QC system
fp = "path_to_csv/your.csv"

# Load the data
green_df = pd.read_csv(fp)

# Test different fiber channels
for fiber_channel in ['Fiber_0', 'Fiber_1', 'Fiber_2', 'Fiber_3']:
    print(f"\n=== Testing {fiber_channel} ===")
    
    # Create QC data for specific fiber channel
    qc_data = FiberPhotometryData(
        fluorescence_df=green_df,
        fiber_channel=fiber_channel,
        background_channel='Background',
        time_channel='ReferenceTime'
    )
    
    # Or use the convenience method:
    # qc_data = FiberPhotometryData.from_csv(fp, fiber_channel)
    
    print(f"Data length: {qc_data.data_length}")
    print(f"Background floor average: {qc_data.floor_ave:.2f}")
    print(f"Fiber channel mean: {qc_data.fiber_data.mean():.2f}")
    
    # Create evaluator and run QC
    evaluator = FiberPhotometryQCEvaluator()
    
    # Run raw metrics
    raw_results = evaluator.evaluate_raw_metrics(qc_data)
    print(f"Raw metrics: {len(raw_results)} checks")
    for result in raw_results:
        status = "PASS" if result.value else "FAIL"
        print(f"  - {result.name}: {status}")
    
    # Run processed metrics
    processed_results = evaluator.evaluate_processed_metrics(qc_data)
    print(f"Processed metrics: {len(processed_results)} checks")
    for result in processed_results:
        status = "PASS" if result.value else "FAIL"
        print(f"  - {result.name}: {status}")



###### Examples of other ways to run specific metrics

evaluator1 = FiberPhotometryQCEvaluator(
    raw_metrics=["DataDurationMetric"],
    processed_metrics=[]
)
result1 = evaluator1.evaluate_raw_metrics(qc_data)[0]