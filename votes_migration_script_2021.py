from functions.assorted_functions import *
from functions.plotting import plot_results


votes_2020 = read_votes_file('votes per settlement 2020.csv', how='settlement')
votes_2021 = read_votes_file('votes per settlement 2021.csv', how='settlement')

votes_2020.drop(9999, inplace=True)
votes_2021.drop(9999, inplace=True)

votes_2020, votes_2021 = preprocess_data(votes_2020, votes_2021, threshold=0.01)
vote_transfers = calculate_transfer_matrix(votes_2020,
                                           votes_2021,
                                           seat_tolerance=1.5,
                                           r2_threshold=0.95)
# save_csv(vote_transfers.round(1), name='votes transfer 2021')
plot_results(vote_transfers, '2020', '2021', transfer_threshold=1)
