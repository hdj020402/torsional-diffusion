def convert_time(time: float) -> tuple[float]:
    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds
