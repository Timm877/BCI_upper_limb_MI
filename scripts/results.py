import pandas as pd


csp = pd.read_csv('./data/offline/intermediate_datafiles/csp_1644917453.727168_filtering_experiments_202100722_MI_atencion_online.csv')
riemann1 = pd.read_csv('./data/offline/intermediate_datafiles/riemann_1644914608.251013_filtering_experiments_202100722_MI_atencion_online.csv')

all = pd.concat([csp, riemann1], ignore_index=True)
all = all.sort_values(by=['test_accuracy'], ascending=False)
print(all.head(15))
all.to_csv('./data/offline/intermediate_datafiles/filtexp_all_sorted_722.csv', index=False)