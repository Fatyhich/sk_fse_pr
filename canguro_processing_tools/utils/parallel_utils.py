from typing import Callable, Any, List
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map


def do_parallel(task_fn: Callable[[Any], Any], arguments: List[Any], n_workers: int, use_tqdm: bool) -> List[Any]:
    assert isinstance(n_workers, int) and n_workers >= 0, f"n_workers must be int >=0, got {n_workers}"
    if n_workers == 0:
        result = []
        for arg in arguments:
            result.append(task_fn(arg))
        return result
    else:
        if use_tqdm:
            return process_map(task_fn, arguments, max_workers=n_workers)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                result = executor.map(task_fn, arguments)
                return [e for e in result]
