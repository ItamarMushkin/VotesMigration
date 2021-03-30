from functions.assorted_functions import *


votes_2020 = read_votes_file('votes per booth 2020.csv', how='kalpi')
votes_2019b = read_votes_file('votes per booth 2019b.csv', how='kalpi')
stable_2020 = read_stable_booths_file('stable_kalp_2020a.csv')

align_consecutive_elections(votes_2019b, votes_2020, stable_2020)

normalize_voters_and_turnout(votes_2019b, votes_2020)

group_minor_parties(votes_2019b)
group_minor_parties(votes_2020)
assert_datasets_aligned(votes_2019b, votes_2020)

transfer_coefficients = solve_transfer_coefficients_by_kalpi(votes_2019b, votes_2020)

