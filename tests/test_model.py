import numpy as np
import pandas as pd
import pytest

from study import model


def small(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"student_id": [f"S{i}" for i in range(n)], "hours": rng.normal(8, 3, n), "prior": rng.normal(60, 12, n), "final": rng.normal(65, 10, n)})


def test_split_covers_every_row_once_and_has_the_right_size():
    d = small(100)
    tr, te = model.train_test_split(d, 0.2, seed=1)
    assert len(te) == 20 and len(tr) == 80
    assert set(tr["student_id"]).isdisjoint(te["student_id"]) and set(tr["student_id"]) | set(te["student_id"]) == set(d["student_id"])


def test_split_is_repeatable_and_changes_with_the_seed():
    d = small(100)
    a1, _ = model.train_test_split(d, 0.2, seed=1)
    a2, _ = model.train_test_split(d, 0.2, seed=1)
    b, _ = model.train_test_split(d, 0.2, seed=2)
    assert a1["student_id"].tolist() == a2["student_id"].tolist() and a1["student_id"].tolist() != b["student_id"].tolist()


def test_the_filler_uses_train_medians_only():
    train = pd.DataFrame({"hours": [1.0, 2.0, 3.0, np.nan], "prior": [10.0, 20.0, 30.0, np.nan]})
    test = pd.DataFrame({"hours": [np.nan, 1000.0], "prior": [np.nan, 5000.0]})
    f = model.MedianFiller().fit(train)
    assert f.medians_ == {"hours": 2.0, "prior": 20.0}
    out = f.transform(test)
    assert out["hours"].tolist() == [2.0, 1000.0] and out["prior"].tolist() == [20.0, 5000.0]
    assert out["hours_was_gap"].tolist() == [True, False]


def test_changing_the_test_rows_cannot_change_a_fill_value():
    d = small(100)
    d.loc[::7, "hours"] = np.nan
    tr, te = model.train_test_split(d, 0.2, seed=3)
    medians_before = model.MedianFiller().fit(tr).medians_
    te2 = te.copy()
    te2["hours"] = te2["hours"].where(te2["hours"].isna(), 99999.0)    # keep the gaps, make every other test value extreme
    te2["prior"] = te2["prior"].where(te2["prior"].isna(), -99999.0)
    assert te["hours"].isna().sum() > 0
    filled_a = model.MedianFiller().fit(tr).transform(te)
    filled_b = model.MedianFiller().fit(tr).transform(te2)
    assert model.MedianFiller().fit(tr).medians_ == medians_before
    assert filled_a.loc[te["hours"].isna(), "hours"].eq(medians_before["hours"]).all()
    assert filled_b.loc[te2["hours"].isna(), "hours"].eq(medians_before["hours"]).all()


def test_transform_does_not_modify_its_input():
    train = pd.DataFrame({"hours": [1.0, np.nan], "prior": [1.0, 2.0]})
    copy = train.copy(deep=True)
    model.MedianFiller().fit(train).transform(train)
    pd.testing.assert_frame_equal(train, copy)


def test_standardiser_gives_mean_0_sd_1_on_its_own_data_and_uses_the_train_numbers_elsewhere():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    s = model.Standardiser().fit(x)
    z = s.transform(x)
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    assert s.transform(np.array([25.0]))[0] == pytest.approx(0)       # 25 is the train mean
    with pytest.raises(ValueError):
        model.Standardiser().fit(np.array([5.0, 5.0, 5.0]))


def test_gradient_descent_matches_least_squares():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 200)
    y = 66 + 7.5 * z + rng.normal(0, 5, 200)
    w0, w1, _ = model.gradient_descent(z, y, 0.1, 500)
    l0, l1 = model.least_squares(z, y)
    assert w0 == pytest.approx(l0, abs=1e-6) and w1 == pytest.approx(l1, abs=1e-6)


def test_gradient_descent_recovers_known_weights_without_noise():
    z = np.linspace(-2, 2, 50)
    y = 3 + 4 * z
    w0, w1, costs = model.gradient_descent(z, y, 0.3, 300)
    assert w0 == pytest.approx(3, abs=1e-6) and w1 == pytest.approx(4, abs=1e-6) and costs[-1] < 1e-10


def test_the_cost_goes_down_every_step_with_a_sensible_rate_and_has_steps_plus_one_points():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 100)
    y = 50 + 5 * z + rng.normal(0, 3, 100)
    _, _, costs = model.gradient_descent(z, y, 0.1, 100)
    assert len(costs) == 101 and all(b <= a + 1e-12 for a, b in zip(costs, costs[1:]))


def test_a_far_too_big_rate_makes_the_cost_explode():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 100)
    y = 50 + 5 * z
    _, _, costs = model.gradient_descent(z, y, 2.5, 30)
    assert costs[-1] > costs[0] * 1e3


def test_a_tiny_rate_is_still_far_from_done_after_a_few_steps():
    z = np.linspace(-2, 2, 50)
    y = 3 + 4 * z
    _, _, costs = model.gradient_descent(z, y, 0.001, 5)
    assert costs[-1] > 0.9 * costs[0]


def test_rmse():
    assert model.rmse([1, 2, 3], [1, 2, 3]) == 0
    assert model.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
