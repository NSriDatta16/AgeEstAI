from __future__ import annotations

AGE_BINS = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-64", "65+"]


def age_to_bin(age: float) -> str:
    """Convert a continuous apparent-age estimate into the UI age bins."""
    age = float(age)
    if age <= 12:
        return "0-12"
    if age <= 19:
        return "13-19"
    if age <= 29:
        return "20-29"
    if age <= 39:
        return "30-39"
    if age <= 49:
        return "40-49"
    if age <= 64:
        return "50-64"
    return "65+"
