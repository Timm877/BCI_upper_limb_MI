import pandas as pd


csp = pd.read_csv('./data/offline/intermediate_datafiles/filtering_experiments_csp.csv')
riemann1 = pd.read_csv('./data/offline/intermediate_datafiles/filtering_experiments_riemann.csv')
riemann2 = pd.read_csv('./data/offline/intermediate_datafiles/second_filtering_experiments_riemann.csv')

all = pd.concat([csp, riemann1,riemann2], ignore_index=True)
all = all.sort_values(by=['test_accuracy'], ascending=False)
print(all.head(15))
all.to_csv('./data/offline/intermediate_datafiles/filtexp_all_sorted.csv', index=False)