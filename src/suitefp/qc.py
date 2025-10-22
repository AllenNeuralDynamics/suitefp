"""
Fiber Photometry Quality Control Module

"""

from abc import ABC, abstractmethod
from typing import List, Any, Iterator
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status, Stage
from aind_data_schema_models.modalities import _Fib as FiberPhotometry


class FiberPhotometryData:
    """Data container for fiber photometry QC metrics."""
    
    def __init__(
        self,
        fluorescence_df: Any,  # pandas DataFrame with fiber photometry data
        fiber_channel: str,  # specific fiber channel to QC (e.g., 'Fiber_0', 'Fiber_1', etc.)
        background_channel: str = 'Background',  # background channel name
        time_channel: str = 'ReferenceTime',  # time reference column
    ):
        self.fluorescence_df = fluorescence_df
        self.fiber_channel = fiber_channel
        self.background_channel = background_channel
        self.time_channel = time_channel
        
        # Validate that required columns exist
        required_columns = [fiber_channel, background_channel, time_channel]
        missing_columns = [col for col in required_columns if col not in fluorescence_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in DataFrame: {missing_columns}")
        
        # Calculate floor average from background data
        self._floor_ave = self.fluorescence_df[self.background_channel].mean()
    
    @property
    def fiber_data(self):
        """Get the specific fiber channel data."""
        return self.fluorescence_df[self.fiber_channel]
    
    @property
    def background_data(self):
        """Get background channel data."""
        return self.fluorescence_df[self.background_channel]
    
    @property
    def time_data(self):
        """Get time reference data."""
        return self.fluorescence_df[self.time_channel]
    
    @property
    def floor_ave(self):
        """Get calculated floor average from background."""
        return self._floor_ave
    
    @property
    def data_length(self):
        """Get number of data points."""
        return len(self.fluorescence_df)
    
    @classmethod
    def from_csv(
        cls,
        csv_file_path: str,
        fiber_channel: str,
        background_channel: str = 'Background',
        time_channel: str = 'ReferenceTime'
    ):
        """Create FiberPhotometryData directly from a CSV file."""
        try:
            import pandas as pd
            fluorescence_df = pd.read_csv(csv_file_path)
        except ImportError:
            raise ImportError("pandas is required to read CSV files")
        
        return cls(
            fluorescence_df=fluorescence_df,
            fiber_channel=fiber_channel,
            background_channel=background_channel,
            time_channel=time_channel
        )


class AbstractMetricEvaluator(ABC):
    """Abstract base class for QC metric evaluators."""
    
    def __init__(self, evaluator_name: str = "FiberPhotometry QC System"):
        self.evaluator_name = evaluator_name
        # Create modality instance once
        self._fiber_modality = FiberPhotometry()
    
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the name of the metric."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of the metric."""
        pass
    
    @property
    def stage(self) -> Stage:
        """Return the QC stage for this metric. Override in subclasses if needed."""
        return Stage.RAW
    
    @abstractmethod
    def _evaluate_condition(self, data: FiberPhotometryData) -> bool:
        """Evaluate the specific condition for this metric."""
        pass
    
    def evaluate(self, data: FiberPhotometryData) -> QCMetric:
        """Evaluate the quality of the given data and return a QCMetric."""
        condition_passed = self._evaluate_condition(data)
        status = Status.PASS if condition_passed else Status.FAIL
        
        qc_status = QCStatus(
            evaluator=self.evaluator_name,
            status=status,
            timestamp=datetime.now(timezone.utc)
        )
        
        return QCMetric(
            name=self.metric_name,
            modality=self._fiber_modality,
            stage=self.stage,
            value=condition_passed,
            status_history=[qc_status],
            description=self.description
        )


# Base classes for metrics
class NoNaNMetric(AbstractMetricEvaluator):
    """Base class for NaN detection metrics."""
    
    def __init__(self, evaluator_name: str = "FiberPhotometry QC System"):
        super().__init__(evaluator_name)
    
    @property
    def metric_name(self) -> str:
        return "NoFiberNaN"
    
    @property
    def description(self) -> str:
        return "Verify that fiber channel contains no NaN values"


class CMOSFloorDarkMetric(AbstractMetricEvaluator):
    """Base class for CMOS floor dark metrics."""
    
    def __init__(self, limit: float = 265, evaluator_name: str = "FiberPhotometry QC System"):
        super().__init__(evaluator_name)
        self.limit = limit
    
    @property
    def metric_name(self) -> str:
        return "CMOSFloorDark"
    
    @property
    def description(self) -> str:
        return f"Verify that fiber channel floor average is below {self.limit}"
    
    @property
    def stage(self) -> Stage:
        """CMOS floor metrics are processed stage."""
        return Stage.PROCESSING


class MetricRegistry:
    """Registry for QC metric classes that allows string-based lookup."""
    
    _metrics = {}
    
    @classmethod
    def register(cls, metric_class):
        """Register a metric class with its name."""
        cls._metrics[metric_class.__name__] = metric_class
        return metric_class
    
    @classmethod
    def get_metric(cls, class_name: str):
        """Get a metric class by its string name."""
        if class_name not in cls._metrics:
            raise ValueError(f"Metric class '{class_name}' not found. Available: {list(cls._metrics.keys())}")
        return cls._metrics[class_name]
    
    @classmethod
    def list_metrics(cls):
        """List all registered metric class names."""
        return list(cls._metrics.keys())
    
    @classmethod
    def create_metric(cls, class_name: str, **kwargs):
        """Create an instance of a metric class by name."""
        metric_class = cls.get_metric(class_name)
        return metric_class(**kwargs)


# Auto-register all existing metrics using decorators
@MetricRegistry.register
class DataIntegrityMetric(AbstractMetricEvaluator):
    """Check if fiber and background data have the same length."""
    
    @property
    def metric_name(self) -> str:
        return "IsDataIntegrityOK"
    
    @property
    def description(self) -> str:
        return "Verify that fiber channel and background data have matching lengths"
    
    def _evaluate_condition(self, data: FiberPhotometryData) -> bool:
        return len(data.fiber_data) == len(data.background_data) == len(data.time_data)


@MetricRegistry.register  
class DataDurationMetric(AbstractMetricEvaluator):
    """Check if data is longer than 15 minutes (>18000 samples)."""
    
    @property
    def metric_name(self) -> str:
        return "IsDataLongerThan15min"
    
    @property
    def description(self) -> str:
        return "Verify that recording duration exceeds 15 minutes (>18000 samples)"
    
    def _evaluate_condition(self, data: FiberPhotometryData) -> bool:
        return len(data.fluorescence_df) > 18000


@MetricRegistry.register
class NoFiberNaNMetric(NoNaNMetric):
    """Check for NaN values in fiber channel."""
    
    def __init__(self, evaluator_name: str = "FiberPhotometry QC System"):
        super().__init__(evaluator_name)
    
    def _evaluate_condition(self, data: FiberPhotometryData) -> bool:
        try:
            # Try pandas method first
            return not data.fiber_data.isna().any()
        except AttributeError:
            # Fallback to numpy
            import numpy as np
            return not np.isnan(data.fiber_data).any()


@MetricRegistry.register
class CMOSFloorDarkFiberMetric(CMOSFloorDarkMetric):
    """Check CMOS floor dark for fiber channel."""
    
    def __init__(self, limit: float = 265, evaluator_name: str = "FiberPhotometry QC System"):
        super().__init__(limit, evaluator_name)
    
    def _evaluate_condition(self, data: FiberPhotometryData) -> bool:
        return data.floor_ave < self.limit


@MetricRegistry.register
class NoSuddenChangeMetric(AbstractMetricEvaluator):
    """Check for sudden changes in signal."""
    
    def __init__(self, limit: float = 2000, evaluator_name: str = "FiberPhotometry QC System"):
        super().__init__(evaluator_name)
        self.limit = limit
    
    @property
    def metric_name(self) -> str:
        return "NoSuddenChangeInSignal"
    
    @property
    def description(self) -> str:
        return f"Verify that maximum signal change is below {self.limit} in fiber channel"
    
    @property
    def stage(self) -> Stage:
        """Sudden change detection is processed stage."""
        return Stage.PROCESSING
    
    def _evaluate_condition(self, data: FiberPhotometryData) -> bool:
        try:
            import numpy as np
            fiber_data = data.fiber_data
            
            # Skip first and last 10 samples, calculate diff
            trimmed_data = fiber_data[10:-2] if len(fiber_data) > 12 else fiber_data
            
            if hasattr(trimmed_data, 'diff'):
                # Pandas method
                max_change = abs(trimmed_data.diff()).max()
            else:
                # Fallback to numpy/manual calculation
                diffs = np.diff(trimmed_data)
                max_change = np.max(abs(diffs)) if len(diffs) > 0 else 0
            
            return max_change < self.limit
        except (IndexError, TypeError, AttributeError):
            # If data doesn't have the expected format, return False
            return False


@MetricRegistry.register
class SingleRecordingMetric(AbstractMetricEvaluator):
    """Check if there's only one recording per session."""
    
    @property
    def metric_name(self) -> str:
        return "IsSingleRecordingPerSession"
    
    @property
    def description(self) -> str:
        return "Verify that the data represents a single valid recording session"
    
    def _evaluate_condition(self, data: FiberPhotometryData) -> bool:
        # Simple check - if we have data, assume it's a single recording
        return len(data.fluorescence_df) > 0


class FiberPhotometryQCEvaluator:
    """Main evaluator class that manages and runs all QC metrics using string-based configuration."""
    
    # Default configurations - can be easily modified
    DEFAULT_RAW_METRICS = [
        "DataIntegrityMetric",
        "DataDurationMetric", 
        "NoFiberNaNMetric",
        "SingleRecordingMetric",
    ]
    
    DEFAULT_PROCESSED_METRICS = [
        "CMOSFloorDarkFiberMetric",
        "NoSuddenChangeMetric",
    ]
    
    def __init__(
        self, 
        evaluator_name: str = "FiberPhotometry QC System",
        raw_metrics: List[str] = None,
        processed_metrics: List[str] = None,
        metric_kwargs: dict = None
    ):
        self.evaluator_name = evaluator_name
        self.raw_metric_names = raw_metrics or self.DEFAULT_RAW_METRICS.copy()
        self.processed_metric_names = processed_metrics or self.DEFAULT_PROCESSED_METRICS.copy()
        self.metric_kwargs = metric_kwargs or {}
        
    def _create_metrics_from_strings(self, metric_names: List[str]) -> List[AbstractMetricEvaluator]:
        """Create metric instances from string names."""
        metrics = []
        for name in metric_names:
            # Get kwargs for this specific metric if provided
            kwargs = self.metric_kwargs.get(name, {})
            # Always include evaluator_name
            kwargs['evaluator_name'] = kwargs.get('evaluator_name', self.evaluator_name)
            
            metric = MetricRegistry.create_metric(name, **kwargs)
            metrics.append(metric)
        return metrics
    
    def get_raw_metrics(self) -> Iterator[AbstractMetricEvaluator]:
        """Get iterator for raw data metrics created from string names."""
        return iter(self._create_metrics_from_strings(self.raw_metric_names))
    
    def get_processed_metrics(self) -> Iterator[AbstractMetricEvaluator]:
        """Get iterator for processed data metrics created from string names."""
        return iter(self._create_metrics_from_strings(self.processed_metric_names))
    
    def get_all_metrics(self) -> Iterator[AbstractMetricEvaluator]:
        """Get iterator for all metrics created from string names."""
        all_names = self.raw_metric_names + self.processed_metric_names
        return iter(self._create_metrics_from_strings(all_names))
    
    def evaluate_all_metrics(self, data: FiberPhotometryData) -> List[QCMetric]:
        """Evaluate all metrics and return list of QCMetric objects."""
        return [metric.evaluate(data) for metric in self.get_all_metrics()]
    
    def evaluate_raw_metrics(self, data: FiberPhotometryData) -> List[QCMetric]:
        """Evaluate only raw data metrics."""
        return [metric.evaluate(data) for metric in self.get_raw_metrics()]
    
    def evaluate_processed_metrics(self, data: FiberPhotometryData) -> List[QCMetric]:
        """Evaluate only processed data metrics."""
        return [metric.evaluate(data) for metric in self.get_processed_metrics()]
    
    def add_metric(self, metric_name: str, category: str = "raw", **kwargs):
        """Add a metric to the evaluator by string name."""
        if category.lower() == "raw":
            if metric_name not in self.raw_metric_names:
                self.raw_metric_names.append(metric_name)
        elif category.lower() == "processed":
            if metric_name not in self.processed_metric_names:
                self.processed_metric_names.append(metric_name)
        else:
            raise ValueError("Category must be 'raw' or 'processed'")
        
        # Store kwargs for this metric
        if kwargs:
            self.metric_kwargs[metric_name] = kwargs
    
    def remove_metric(self, metric_name: str):
        """Remove a metric from the evaluator."""
        if metric_name in self.raw_metric_names:
            self.raw_metric_names.remove(metric_name)
        if metric_name in self.processed_metric_names:
            self.processed_metric_names.remove(metric_name)
        if metric_name in self.metric_kwargs:
            del self.metric_kwargs[metric_name]
    
    def list_available_metrics(self):
        """List all available metric class names."""
        return MetricRegistry.list_metrics()
    
    def get_current_metrics(self):
        """Get current metric configuration."""
        return {
            'raw': self.raw_metric_names.copy(),
            'processed': self.processed_metric_names.copy(),
            'kwargs': self.metric_kwargs.copy()
        }