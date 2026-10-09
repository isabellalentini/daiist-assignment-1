"""
Shared data preparation for the hotel cancellation model.

Both train.ipynb and app.py import from here, so a booking is prepared the
same way whether it is used for training or for a prediction in the app.
"""

import numpy as np
import pandas as pd

MAX_CHILDREN = 3

# The columns the model sees, grouped by how preprocessing treats them.
NUMERIC_FEATURES = [
    "no_of_adults",
    "no_of_children",
    "avg_price_per_room",
    "no_of_special_requests",
    "log_lead_time",
    "log_total_nights",
    "weekend_share",
    "arrival_month_sin",
    "arrival_month_cos",
    "booking_month_sin",
    "booking_month_cos",
]
BINARY_FEATURES = [
    "required_car_parking_space",
    "repeated_guest",
    "invalid_date",
    "zero_nights",
    "is_free",
    "has_prev_cancellation",
    "has_prev_stay",
]
CATEGORICAL_FEATURES = [
    "type_of_meal_plan",
    "room_type_reserved",
    "market_segment_type",
    "arrival_weekday",
]


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


def _month_on_circle(month: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Place months 1 to 12 on a circle so December sits next to January."""
    angle = 2 * np.pi * (month - 1) / 12
    return np.sin(angle), np.cos(angle)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from a cleaned table. Uses each row on its own,
    so nothing is learned from the data and it is safe on any split."""
    df = df.copy()

    # Skewed counts: log(1 + x) shrinks the long tail and handles x = 0.
    df["log_lead_time"] = np.log1p(df["lead_time"])
    total_nights = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["log_total_nights"] = np.log1p(total_nights)

    # Share of the stay that falls on the weekend (0 for zero-night bookings).
    df["weekend_share"] = (df["no_of_weekend_nights"] / total_nights.replace(0, np.nan)).fillna(0)

    # Seasonality of the arrival, and of the moment the booking was made.
    df["arrival_month_sin"], df["arrival_month_cos"] = _month_on_circle(df["arrival"].dt.month)
    booking_date = df["arrival"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["booking_month_sin"], df["booking_month_cos"] = _month_on_circle(booking_date.dt.month)

    df["arrival_weekday"] = df["arrival"].dt.day_name()

    # Almost everyone has 0 here, so a yes/no flag is more stable than the count.
    df["has_prev_cancellation"] = (df["no_of_previous_cancellations"] > 0).astype(int)
    df["has_prev_stay"] = (df["no_of_previous_bookings_not_canceled"] > 0).astype(int)

    return df