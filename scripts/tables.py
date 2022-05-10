
from tabulate import tabulate
from texttable import Texttable

import latextable

import pandas as pd

df1 = pd.read_csv("results/intermediate_datafiles/openloop/X01_ML/csp_STD_multiclass.csv")
df2 = pd.read_csv("results/intermediate_datafiles/openloop/X02_ML/csp_STD_multiclass.csv")
df3 = pd.read_csv("results/intermediate_datafiles/openloop/X03_ML/csp_STD_multiclass.csv")
df4 = pd.read_csv("results/intermediate_datafiles/openloop/X04_ML/csp_STD_multiclass.csv")
df5 = pd.read_csv("results/intermediate_datafiles/openloop/X05_ML/csp_STD_multiclass.csv")
df6 = pd.read_csv("results/intermediate_datafiles/openloop/X06_ML/csp_STD_multiclass.csv")
df7 = pd.read_csv("results/intermediate_datafiles/openloop/X07_ML/csp_STD_multiclass.csv")
df8 = pd.read_csv("results/intermediate_datafiles/openloop/X08_ML/csp_STD_multiclass.csv")
df9 = pd.read_csv("results/intermediate_datafiles/openloop/X09_ML/csp_STD_multiclass.csv")



rows = [['Frequency band/ Subject', 'X01','X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08', 'X09'],
        ['CSP + sLDA', df1.loc[0,'final_val_accuracy'], df2.loc[0,'final_val_accuracy'],
        df3.loc[0,'final_val_accuracy'],df4.loc[0,'final_val_accuracy'], df5.loc[2,'final_val_accuracy'],
        df6.loc[0,'final_val_accuracy'],df7.loc[2,'final_val_accuracy'], df8.loc[0,'final_val_accuracy'],
        df9.loc[1,'final_val_accuracy'],
             ],
        ['CSP + SVM',df1.loc[1,'final_val_accuracy'], df2.loc[1,'final_val_accuracy'],
        df3.loc[1,'final_val_accuracy'],df4.loc[1,'final_val_accuracy'], df5.loc[1,'final_val_accuracy'],
        df6.loc[1,'final_val_accuracy'],df7.loc[0,'final_val_accuracy'], df8.loc[2,'final_val_accuracy'],
        df9.loc[0,'final_val_accuracy'],
             ],
        ['CSP + RF', df1.loc[2,'final_val_accuracy'], df2.loc[2,'final_val_accuracy'],
        df3.loc[2,'final_val_accuracy'],df4.loc[2,'final_val_accuracy'], df5.loc[0,'final_val_accuracy'],
        df6.loc[2,'final_val_accuracy'],df7.loc[1,'final_val_accuracy'], df8.loc[1,'final_val_accuracy'],
        df9.loc[2,'final_val_accuracy'],
        ]
        ]

#,  'X04', 'X05', 'X06', 'X07', 'X08', 'X09'
table = Texttable()
table.set_cols_align(["c"] * 10)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

import numpy as np
slda = np.array([df1.loc[0,'final_val_accuracy'], df2.loc[0,'final_val_accuracy'],
        df3.loc[0,'final_val_accuracy'],df4.loc[0,'final_val_accuracy'], df5.loc[2,'final_val_accuracy'],
        df6.loc[0,'final_val_accuracy'],df7.loc[2,'final_val_accuracy'], df8.loc[0,'final_val_accuracy'],
        df9.loc[1,'final_val_accuracy']]).mean()

svm = np.array([df1.loc[1,'final_val_accuracy'], df2.loc[1,'final_val_accuracy'],
        df3.loc[1,'final_val_accuracy'],df4.loc[1,'final_val_accuracy'], df5.loc[1,'final_val_accuracy'],
        df6.loc[1,'final_val_accuracy'],df7.loc[0,'final_val_accuracy'], df8.loc[2,'final_val_accuracy'],
        df9.loc[0,'final_val_accuracy']]).mean()
rf = np.array([df1.loc[2,'final_val_accuracy'], df2.loc[2,'final_val_accuracy'],
        df3.loc[2,'final_val_accuracy'],df4.loc[2,'final_val_accuracy'], df5.loc[0,'final_val_accuracy'],
        df6.loc[2,'final_val_accuracy'],df7.loc[1,'final_val_accuracy'], df8.loc[1,'final_val_accuracy'],
        df9.loc[2,'final_val_accuracy']]).mean()

print(slda)
print(svm)
print(rf)
'''

from tabulate import tabulate
from texttable import Texttable

import latextable

import pandas as pd

df1_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X01_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df1_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X01_WithSPFilt2\multiclass_riemann_multiclass.csv")

df2_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X02_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df2_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X02_WithSPFilt2\multiclass_riemann_multiclass.csv")

df3_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X03_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df3_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X03_WithSPFilt2\multiclass_riemann_multiclass.csv")

df4_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X04_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df4_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X04_WithSPFilt2\multiclass_riemann_multiclass.csv")

df5_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X05_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df5_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X05_WithSPFilt2\multiclass_riemann_multiclass.csv")

df6_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X06_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df6_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X06_WithSPFilt2\multiclass_riemann_multiclass.csv")

df7_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X07_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df7_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X07_WithSPFilt2\multiclass_riemann_multiclass.csv")

df8_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X08_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df8_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X08_WithSPFilt2\multiclass_riemann_multiclass.csv")

df9_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X09_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df9_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X09_WithSPFilt2\multiclass_riemann_multiclass.csv")

rows = [['Filter order / Subject', 'X01','X02', 'X03', 'X04', 'X05','X06', 'X07', 'X08', 'X09'],
        ['Butter filtfilt', df1_norm.loc[0,'final_val_accuracy'], df2_norm.loc[0,'final_val_accuracy'],
        df3_norm.loc[0,'final_val_accuracy'],df4_norm.loc[0,'final_val_accuracy'],df5_norm.loc[0,'final_val_accuracy'],
        df6_norm.loc[0,'final_val_accuracy'],df7_norm.loc[0,'final_val_accuracy'],df8_norm.loc[0,'final_val_accuracy'],
        df9_norm.loc[0,'final_val_accuracy'],
             ],
        ['State space filt',df1_SP.loc[0,'final_val_accuracy'], df2_SP.loc[0,'final_val_accuracy'],
        df3_SP.loc[0,'final_val_accuracy'],df4_SP.loc[0,'final_val_accuracy'],df5_SP.loc[0,'final_val_accuracy'],
        df6_SP.loc[0,'final_val_accuracy'],df7_SP.loc[0,'final_val_accuracy'],df8_SP.loc[0,'final_val_accuracy'],
        df9_SP.loc[0,'final_val_accuracy']
             ]
        ]

#,  'X04', 'X05', 'X06', 'X07', 'X08', 'X09'
table = Texttable()
table.set_cols_align(["c"] * 10)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))
import numpy as np
norm = np.array([df1_norm.loc[0,'final_val_accuracy'], df2_norm.loc[0,'final_val_accuracy'],
        df3_norm.loc[0,'final_val_accuracy'],df4_norm.loc[0,'final_val_accuracy'],df5_norm.loc[0,'final_val_accuracy'],
        df6_norm.loc[0,'final_val_accuracy'],df7_norm.loc[0,'final_val_accuracy'],df8_norm.loc[0,'final_val_accuracy'],
        df9_norm.loc[0,'final_val_accuracy']]).mean()

sp = np.array([df1_SP.loc[0,'final_val_accuracy'], df2_SP.loc[0,'final_val_accuracy'],
        df3_SP.loc[0,'final_val_accuracy'],df4_SP.loc[0,'final_val_accuracy'],df5_SP.loc[0,'final_val_accuracy'],
        df6_SP.loc[0,'final_val_accuracy'],df7_SP.loc[0,'final_val_accuracy'],df8_SP.loc[0,'final_val_accuracy'],
        df9_SP.loc[0,'final_val_accuracy']]).mean()
print(norm)
print(sp)
'''