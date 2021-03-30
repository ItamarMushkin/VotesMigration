import pandas as pd
import numpy as np
import os
import cvxpy as cvx
from matplotlib import pyplot as plt

from datetime import datetime


from column_names import ColumnNames

ENCODING = 'iso-8859-8'
# FOLDER_PATH = r'C:\Users\Asus\Google Drive\Votes Migration 2020'
FOLDER_PATH = 'data files'
KNESSET_SIZE = 120


def read_votes_file(filename, how: str = 'settlement'):
    if how == 'settlement':
        return _read_votes_file_settlement(filename)
    elif how == 'kalpi':
        return _read_votes_file_kalpi(filename)
    else:
        raise ValueError


def _read_votes_file_kalpi(filename):
    file_path = os.path.join(FOLDER_PATH, filename)
    df = pd.read_csv(file_path, encoding=ENCODING)

    df.set_index(ColumnNames.index_columns, inplace=True)
    df.drop(ColumnNames.additional_oversight_columns, inplace=True, errors='ignore', axis=1)
    df.dropna(how='all', axis=1, inplace=True)
    assert (df.notna().all().all())

    return df


def _read_votes_file_settlement(filename):
    file_path = os.path.join(FOLDER_PATH, filename)
    df = pd.read_csv(file_path, encoding=ENCODING)

    df.set_index(ColumnNames.settlement_column, inplace=True)
    df.drop(ColumnNames.additional_oversight_columns, inplace=True, errors='ignore', axis=1)
    df.dropna(how='all', axis=1, inplace=True)
    assert (df.notna().all().all())

    return df


def read_stable_booths_file(filename):
    stable_booths = pd.read_csv(os.path.join(FOLDER_PATH, filename), encoding=ENCODING)
    stable_booths.columns = ['סמל ישוב', 'שם ישוב', 'קלפי']
    stable_booths['קלפי'] = stable_booths['קלפי'].astype(float)
    stable_booths = stable_booths.set_index(['סמל ישוב', 'קלפי']).index

    return stable_booths


def preprocess_data(df1, df2, threshold=0.01):
    align_consecutive_elections(df1, df2)
    normalize_voters_and_turnout(df1, df2)
    group_minor_parties(df1, threshold)
    group_minor_parties(df2, threshold)
    assert_datasets_aligned(df1, df2, threshold)
    df1 = df1[parties_with_other(df1, threshold)]
    df2 = df2[parties_with_other(df2, threshold)]
    return df1, df2


def align_consecutive_elections(df1, df2, stable_booths=None):
    if stable_booths is not None:
        df1.drop(df1.index.difference(stable_booths), inplace=True)
        df2.drop(df2.index.difference(stable_booths), inplace=True)
    df2.drop(df2.index.difference(df1.index), inplace=True)
    df1.drop(df1.index.difference(df2.index), inplace=True)
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    pd.testing.assert_index_equal(df1.index, df2.index)


def party_columns(df):
    return df.columns.difference(pd.Index(ColumnNames.voters_count_columns+['שם ישוב', 'turnout_pct']))


def normalize_voters_and_turnout(df1, df2):

    df1['increased turnout'] = df2[ColumnNames.total_votes].sub(df1[ColumnNames.total_votes]).clip(lower=0)
    df2['decreased turnout'] = df1[ColumnNames.total_votes].sub(df2[ColumnNames.total_votes]).clip(lower=0)

    df1.drop(df1.columns[df1.sum().eq(0)], axis=1, inplace=True)
    df2.drop(df2.columns[df2.sum().eq(0)], axis=1, inplace=True)

    pd.testing.assert_series_equal(
        df1[party_columns(df1)].sum(axis=1),
        df2[party_columns(df2)].sum(axis=1)
    )


def major_parties(df, threshold=0.01):
    party_votes = df[party_columns(df)].sum()
    party_votes_pct = party_votes / party_votes.sum()
    _major_parties = party_votes_pct[party_votes_pct > threshold].index

    return _major_parties


def group_minor_parties(df, threshold=0.01):
    minor_parties = party_columns(df).difference(major_parties(df, threshold)).difference(
        pd.Index(['increased turnout', 'decreased turnout']))
    df['other'] = df[minor_parties].sum(axis=1)
    df.drop(minor_parties, axis=1, inplace=True)


def parties_with_other(df, threshold):
    parties = major_parties(df, threshold)
    parties = parties.append(pd.Index(['other'])) if 'other' not in parties else parties
    parties = parties.append(pd.Index(['decreased turnout'])) if (
                'decreased turnout' not in parties and 'decreased turnout' in df.columns) else parties
    parties = parties.append(pd.Index(['increased turnout'])) if (
                'increased turnout' not in parties and 'increased turnout' in df.columns) else parties
    return parties


def assert_datasets_aligned(df1, df2, threshold):
    pd.testing.assert_series_equal(
        df1.drop(['בזב', 'מצביעים', 'פסולים', 'כשרים'], axis=1).sum(axis=1),
        df2.drop(['בזב', 'מצביעים', 'פסולים', 'כשרים'], axis=1).sum(axis=1)
    )

    pd.testing.assert_series_equal(
        df1[parties_with_other(df1, threshold)].sum(axis=1),
        df2[parties_with_other(df2, threshold)].sum(axis=1)
    )


def calculate_transfer_matrix(df1,
                              df2,
                              seat_tolerance,
                              r2_threshold
                              ):
    transfer_coefficients, rmse = solve_transfer_coefficients_by_settlement(df1, df2)
    transfer_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, data=transfer_coefficients)
    r2 = validate_transfer_matrix(df1, df2, transfer_matrix, rmse, r2_threshold)
    print('r2 value is {}'.format(r2))
    vote_transfers = transfer_matrix.apply(lambda x: x * (KNESSET_SIZE * df1.sum() / df1.sum().sum()))
    validate_approximate_votes(vote_transfers, seat_tolerance=seat_tolerance)
    return vote_transfers


def solve_transfer_coefficients_by_settlement(df1, df2):
    coefficients = cvx.Variable(shape=(len(df1.columns), len(df2.columns)))
    constraints = [0 <= coefficients, coefficients <= 1, cvx.sum(coefficients, axis=1) == 1]

    rmse = cvx.norm(df1.values @ coefficients - df2.values, 'fro')
    objective = cvx.Minimize(rmse)
    prob = cvx.Problem(objective, constraints)

    rmse = prob.solve(verbose=True, solver='SCS', max_iters=20000)
    return coefficients.value, rmse


def solve_transfer_coefficients_by_kalpi(df1, df2):
    coefficients = cvx.Variable(shape=(len(df1.columns), len(df2.columns)))
    constraints = [0 <= coefficients, coefficients <= 1, cvx.sum(coefficients, axis=1) == 1]

    mse = cvx.norm(
        df1.groupby('שם ישוב').sum().values @ coefficients - df2.groupby('שם ישוב').sum().values, 'fro')
    objective = cvx.Minimize(mse)
    prob = cvx.Problem(objective, constraints)

    mse = prob.solve(verbose=True, solver='SCS', max_iters=20000)
    return coefficients.value, mse


def validate_transfer_matrix(df1, df2, transfer_matrix, rmse, r2_threshold):
    s1 = np.linalg.norm(df1 @ transfer_matrix - df2)
    assert np.isclose(s1, rmse)
    party_sizes = df2.sum() / df2.sum().sum()
    base_transfer_matrix = pd.DataFrame(data=[party_sizes]*len(df1.columns), index=df1.columns)
    assert np.allclose((df1 @ base_transfer_matrix - df2).sum(), 0)
    s0 = np.linalg.norm(df1 @ base_transfer_matrix - df2)
    r2 = 1-(s1**2)/(s0**2)
    assert r2 > r2_threshold
    return r2


ACTUAL_SEATS = {
    'מחל': 30,
    'פה': 17,
    'שס': 9,
    'כן': 8,
    'ב': 7,
    'אמת': 7,
    'ג': 7,
    'ל': 7,
    'ט': 6,
    'ודעם': 6,
    'ת': 6,
    'מרצ': 6,
    'עם': 4
}


def validate_approximate_votes(vote_transfers, actual_seats=None, seat_tolerance=1.5):
    actual_seats = actual_seats or ACTUAL_SEATS
    party_sizes_approx = (vote_transfers.sum() * KNESSET_SIZE / vote_transfers.sum().drop(
        ColumnNames.non_party_columns).sum()).drop(ColumnNames.non_party_columns).round(1).sort_values(ascending=False)
    pd.testing.assert_series_equal(party_sizes_approx[actual_seats.keys()].sort_index(),
                                   pd.Series(actual_seats, dtype=float).sort_index(),
                                   atol=seat_tolerance)


def display_df(_df):  # Thanks, Dean Langsam!
    _display_df = _df.join(pd.DataFrame(columns=_df.index.difference(_df.columns)))
    _display_df = _display_df.append(pd.DataFrame(index=_display_df.columns.difference(_display_df.index),
                                                  data=0,
                                                  columns=_display_df.columns)).fillna(0)
    _display_df = _display_df.sort_index(axis=0).sort_index(axis=1).T.round(3)
    return _display_df.round(2).style.background_gradient(cmap=plt.get_cmap('Accent_r'))


def save_csv(df, name):
    now = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H-%M-%S")
    # df.to_csv('{}_{}.csv'.format(name, now, encoding=ENCODING))
    df.to_csv(os.path.join(FOLDER_PATH, '{}_{}.csv').format(name, now, encoding=ENCODING))


def save_excel(coefficients,
               transfers,
               name,
               sheet_names=('transfer coefficients', 'vote transfers')):
    now = datetime.now()
    coefficients.to_excel('{}_{}.xlsx'.format(name, now), encoding=ENCODING, sheet_name=sheet_names[0])
    transfers.to_excel('{}_{}.xlsx'.format(name, now), encoding=ENCODING, sheet_name=sheet_names[1])
