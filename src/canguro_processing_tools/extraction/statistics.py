from typing import TypeVar, Generic, List, Optional, Tuple, Any, Dict, Union
from abc import ABC, abstractmethod
from pathlib import Path
from functools import partial
from canguro_processing_tools.extraction.extraction_handle import ExtractionHandle
from canguro_processing_tools.utils.parallel_utils import do_parallel
from canguro_processing_tools.utils.str_utils import seconds_to_human_readble_time


SAMPLE_TYPE = TypeVar("SAMPLE_TYPE")
POOL_TYPE = TypeVar("POOL_TYPE")


class Ratio:

    def __init__(self, numerator: float, denominator: float) -> None:
        self._numerator = numerator
        self._denominator = denominator

    @property
    def numerator(self) -> float:
        return self._numerator
    
    @property
    def denominator(self) -> float:
        return self._denominator
    
    def __str__(self) -> str:
        return f"{self._numerator} / {self._denominator}"


class AbstractExtractionStatistic(ABC, Generic[SAMPLE_TYPE, POOL_TYPE]):

    def __init__(self,
                 statistic_id: str,
                 statistic_name: str) -> None:
        self._statistic_id = statistic_id
        self._staticstic_name = statistic_name

    @property
    def statistic_id(self) -> str:
        return self._statistic_id
    
    @property
    def statistic_name(self) -> str:
        return self._staticstic_name
    
    @abstractmethod
    def process_sample(self, extraction_handle: ExtractionHandle) -> Optional[SAMPLE_TYPE]:
        pass

    @abstractmethod
    def pool(self, samples: List[Optional[SAMPLE_TYPE]]) -> POOL_TYPE:
        pass

    @abstractmethod
    def build_sample_message(self, extraction_dir: Union[str, Path], sample: SAMPLE_TYPE) -> str:
        pass

    @abstractmethod
    def build_pooling_message(self, pooled_value: POOL_TYPE) -> str:
        pass


class OdometryTimeStatistic(AbstractExtractionStatistic[float, float]):

    ID = "odometry_time"
    NAME = "Odometry time"

    def __init__(self, fps: float) -> None:
        super(OdometryTimeStatistic, self).__init__(OdometryTimeStatistic.ID, OdometryTimeStatistic.NAME)
        self._fps = fps
        
    def process_sample(self, extraction_handle: ExtractionHandle) -> Optional[float]:
        return len(extraction_handle.odometry_timestamps) / self._fps
    
    def pool(self, samples: List[Optional[float]]) -> float:
        return sum(samples)

    def build_sample_message(self, extraction_dir: Union[str, Path], sample: float) -> str:
        return f"{extraction_dir} : {seconds_to_human_readble_time(sample)}"

    def build_pooling_message(self, pooled_value: float) -> str:
        return seconds_to_human_readble_time(pooled_value)


class IncompleteCountStatistic(AbstractExtractionStatistic[Tuple[Ratio, float], Tuple[Ratio, float]]):

    ID = "incomplete_count"
    NAME = "Extraction incomplete"

    def __init__(self, fps: float) -> None:
        super(IncompleteCountStatistic, self).__init__(IncompleteCountStatistic.ID, IncompleteCountStatistic.NAME)
        self._fps = fps
        
    def process_sample(self, extraction_handle: ExtractionHandle) -> Optional[Tuple[Ratio, float]]:
        n_frames = len(extraction_handle.frames_timestamps)
        n_poses = len(extraction_handle.odometry_timestamps)        
        if n_frames != n_poses:
            return Ratio(n_poses, n_frames), (n_frames - n_poses) / self._fps
        return None

    
    def pool(self, samples: List[Optional[Tuple[Ratio, float]]]) -> Tuple[Ratio, float]:
        incomplete_samples = [e for e in samples if e is not None]
        incomplete_fraction = Ratio(len(incomplete_samples), len(samples))
        lost_time = sum([e[1] for e in incomplete_samples])
        return incomplete_fraction, lost_time
    
    def build_sample_message(self, extraction_dir: Union[str, Path], sample: Tuple[Ratio, float]) -> str:
        return f"{extraction_dir} : {seconds_to_human_readble_time(sample[1])} ({str(sample[0])} frames)"

    def build_pooling_message(self, pooled_value: Tuple[Ratio, float]) -> str:
        return f"{seconds_to_human_readble_time(pooled_value[1])} ({str(pooled_value[0])} extractions)"


class NoMetadataStatistic(AbstractExtractionStatistic[bool, int]):

    ID = "no_metadata"
    NAME = "No metadata.json"

    def __init__(self) -> None:
        super(NoMetadataStatistic, self).__init__(NoMetadataStatistic.ID, NoMetadataStatistic.NAME)
        
    def process_sample(self, extraction_handle: ExtractionHandle) -> Optional[bool]:
        metadata_file = extraction_handle.dir_path / "metadata.json"
        if metadata_file.is_file():
            return None
        return True
    
    def pool(self, samples: List[Optional[Tuple[bool]]]) -> int:
        return sum([e for e in samples if e is not None])
    
    def build_sample_message(self, extraction_dir: Union[str, Path], sample: float) -> str:
        return f"{extraction_dir}"

    def build_pooling_message(self, pooled_value: float) -> str:
        return f"{pooled_value} extractions"


class OldNamingStatistic(AbstractExtractionStatistic[bool, int]):

    ID = "old_naming"
    NAME = "Old naming"

    def __init__(self) -> None:
        super(OldNamingStatistic, self).__init__(OldNamingStatistic.ID, OldNamingStatistic.NAME)
        
    def process_sample(self, extraction_handle: ExtractionHandle) -> Optional[bool]:
        if extraction_handle.dir_path.name.startswith("2024"):
            return None
        return True
    
    def pool(self, samples: List[Optional[Tuple[bool]]]) -> int:
        return sum([e for e in samples if e is not None])
    
    def build_sample_message(self, extraction_dir: Union[str, Path], sample: float) -> str:
        return f"{extraction_dir}"

    def build_pooling_message(self, pooled_value: float) -> str:
        return f"{pooled_value} extractions"


def _single_extraction_statistics(extraction: ExtractionHandle,
                                  statistics: List[AbstractExtractionStatistic]):
    result = {}
    for stat in statistics:
        stat_value = stat.process_sample(extraction)
        result[stat.statistic_id] = stat_value
    return extraction.dir_path, result


def _pool_values(statistic: AbstractExtractionStatistic,
                 samples_values: List[Tuple[Path, Dict[str, Any]]]) -> Any:
    return statistic.pool([e[1][statistic.statistic_id] for e in samples_values if statistic.statistic_id in e[1]])


def collect_statistics(extractions: List[ExtractionHandle],
                       statistics: List[AbstractExtractionStatistic],
                       n_workers: int = 0,
                       use_tqdm: bool = False) -> Tuple[List[Tuple[Path, Dict[str, Any]]],
                                                        Dict[str, Any]]:
    task_fn = partial(_single_extraction_statistics,
                      statistics=statistics)
    sampels_values = do_parallel(task_fn, extractions,
                                 n_workers=n_workers, use_tqdm=use_tqdm)
    
    pooled_values = {e.statistic_id: _pool_values(e, sampels_values) for e in statistics}

    return sampels_values, pooled_values
