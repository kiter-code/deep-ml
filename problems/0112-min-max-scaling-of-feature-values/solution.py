def min_max(x: list[float]) -> list[float]:
    min_val = min(x)
    max_val = max(x)

    if min_val == max_val:
        return [0.0] * len(x)

    result = []

    for i in x:
        result.append((i - min_val) / (max_val - min_val))

    return result