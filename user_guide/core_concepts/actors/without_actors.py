import flytekit as fl


@fl.task(requests=fl.Resources(mem="1Gi"))
def add_numbers(a: float, b: float) -> float:
    return a + b


@fl.task(container_image="myrepo/myimage-with-scipy:latest")
def calculate_distance(point_a: list[int], point_b: list[int]) -> float:
    from scipy.spatial.distance import euclidean
    return euclidean(point_a, point_b)


@fl.task(cache=True, cache_version="v1")
def is_even(number: int) -> bool:
    return number % 2 == 0


@fl.workflow
def distance_add_wf(point_a: list[int], point_b: list[int]) -> float:
    distance = calculate_distance(point_a=point_a, point_b=point_b)
    return add_numbers(a=distance, b=1.5)


@fl.workflow
def is_even_wf(point_a: list[int]) -> list[bool]:
    return fl.map_task(is_even)(number=point_a)
