import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import plotly.plotly as py
import plotly.offline as py
from matplotlib.colors import to_rgba

plt.style.use('fivethirtyeight')
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'


def plot_results(vote_transfers: pd.DataFrame,
                 name1: str,
                 name2: str,
                 transfer_threshold: float
                 ):
    vote_transfers = vote_transfers.loc[vote_transfers.ge(transfer_threshold).any(axis=1),
                                        vote_transfers.ge(transfer_threshold).any(axis=0)]

    links = np.where(vote_transfers.values >= transfer_threshold)
    labels = (vote_transfers.index.to_list() + vote_transfers.columns.to_list())
    # colors = ['black', 'brown', 'orange', 'blue', 'green', 'red', 'pink', 'purple', 'cyan', 'magenta'] * 2
    color_coding = {'ג': 'black',
                    'שס': 'yellow',
                    'מחל': 'blue',
                    'ט': 'orange',
                    'טב': 'orange',
                    'ב': 'cyan',
                    'ת': 'yellow',
                    'ל': 'pink',
                    'פה': 'purple',
                    'כן': 'magenta',
                    'אמת': 'red',
                    'מרצ': 'green',
                    'עם': 'green',
                    'ודעם': 'orange',
                    'decreased turnout': 'grey',
                    'other': 'black'
                    }
    colors = [color_coding.get(party, 'grey') for party in labels]

    data = dict(
        type='sankey',
        node=dict(pad=15,
                  thickness=20,
                  line=dict(color="black", width=0.5),
                  label=[(s[::-1] if 'o' not in s else s) for s in labels],
                  color=colors
                  ),
        link=dict(source=links[0],
                  target=links[1] + max(links[0]) + 1,
                  value=[vote_transfers.iloc[f[0], f[1]] for f in zip(links[0], links[1])],
                  color=['rgba'+str(to_rgba(colors[i], alpha=0.3)) for i in links[0]],

                  ),
        orientation='h'
    )

    layout = dict(
        title="Shift in votes between parties, from {} to {} elections".format(name1, name2),
        font=dict(size=14)
    )

    fig = dict(data=[data], layout=layout)
    py.iplot(fig)
