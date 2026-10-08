"""
Shared data preparation for the hotel cancellation model.

Both train.ipynb and app.py import from here, so a booking is prepared the
same way whether it is used for training or for a prediction in the app.
"""

import pandas as pd

MAX_CHILDREN = 3


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data problems, add quality flags, and build the target."""
    df = df.copy()

    # 1. 29 Feb 2018 does not exist (2018 is not a leap year). Flag those
    #    rows, then move them to 1 March, the day after 28 February.
    feb29 = (
        (df["arrival_year"] == 2018)
        & (df["arrival_month"] == 2)
        & (df["arrival_date"] == 29)
    )
    df["invalid_date"] = feb29.astype(int)
    df.loc[feb29, "arrival_month"] = 3
    df.loc[feb29, "arrival_date"] = 1

    # 2. One real date column, used for the time split and date features.
    df["arrival"] = pd.to_datetime(
        {
            "year": df["arrival_year"],
            "month": df["arrival_month"],
            "day": df["arrival_date"],
        }
    )

    # 3. Bookings with no nights at all (possibly day use).
    total_nights = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["zero_nights"] = (total_nights == 0).astype(int)

    # 4. Free rooms (Complementary bookings, plus some Online ones).
    df["is_free"] = (df["avg_price_per_room"] == 0).astype(int)

    # 5. 9 and 10 children are almost certainly typing errors; the next
    #    largest value in the data is 3.
    df["no_of_children"] = df["no_of_children"].clip(upper=MAX_CHILDREN)

    # 6. Target: 1 = canceled. New bookings in the app have no status yet.
    if "booking_status" in df.columns:
        df["y"] = (df["booking_status"] == "Canceled").astype(int)

    return df
