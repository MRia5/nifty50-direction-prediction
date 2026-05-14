import pandas as pd

from financial_ml_project.data import make_next_day_direction


def test_make_next_day_direction():
    nifty = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Close": [100.0, 101.0, 99.0],
        }
    )

    target = make_next_day_direction(nifty)

    assert target["target"].iloc[0] == 1
    assert target["target"].iloc[1] == 0
    assert pd.isna(target["target"].iloc[2])
