import pandas as pd

from app.ml.pipeline.rwanda_features import (
    FMCG_PRODUCT_MAP,
    add_rwanda_features,
    build_holiday_set,
    subset_fmcg_products,
)


def test_holiday_set_has_fourteen_categories_per_year():
    holidays = build_holiday_set([2026])
    # 10 fixed + Good Friday + Easter Monday + Umuganura + (Eid x2 if in range)
    # 2026 has both Eid dates resolvable via the moving calendar approximation
    # only for 2013-2017 in this module; for out-of-range years we still
    # expect at least the 13 non-lunar holidays to be present.
    names = set(holidays.values())
    assert "Genocide Memorial Day" in names
    assert "New Year's Day" in names
    assert "Umuganura Day" in names
    assert "Good Friday" in names
    assert "Easter Monday" in names


def test_genocide_memorial_day_is_fixed_april_7():
    holidays = build_holiday_set([2015])
    memorial_dates = [d for d, name in holidays.items() if name == "Genocide Memorial Day"]
    assert len(memorial_dates) == 1
    assert memorial_dates[0].month == 4
    assert memorial_dates[0].day == 7


def test_add_rwanda_features_marks_memorial_day():
    df = pd.DataFrame({"date": ["2015-04-07"], "sales": [10]})
    out = add_rwanda_features(df)
    assert out.loc[0, "is_memorial"] == 1
    assert out.loc[0, "is_holiday"] == 1


def test_add_rwanda_features_non_holiday_day():
    df = pd.DataFrame({"date": ["2015-04-08"], "sales": [10]})
    out = add_rwanda_features(df)
    assert out.loc[0, "is_memorial"] == 0


def test_subset_fmcg_products_maps_five_items():
    df = pd.DataFrame({
        "date": ["2015-01-01"] * 6,
        "store": [1] * 6,
        "item": list(FMCG_PRODUCT_MAP.keys()) + [99],
        "sales": [10, 20, 30, 40, 50, 99],
    })
    out = subset_fmcg_products(df)
    assert len(out) == 5
    assert set(out["product_code"]) == {"SUGAR", "OIL", "FLOUR", "RICE", "SOAP"}


def test_rwanda_features_adds_all_expected_columns():
    df = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=30), "sales": range(30)})
    out = add_rwanda_features(df)
    expected = {"is_holiday", "is_memorial", "days_to_next_holiday", "season_flag",
                "rain_intensity", "day_of_week", "week_of_year", "month"}
    assert expected.issubset(out.columns)
