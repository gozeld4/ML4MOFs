
import pandas as pd
data = pd.read_csv('combined_data_frame_all.csv')
data_1inorg_1edge = data.iloc[1:3761]
data_1inor_1org_1edge = data.iloc[3762:5995]
data_2inor_1edge = data.iloc[5996:6946]

data_1inor_1org_1edge.to_csv('combined_data_frame_1inorg_1edge.csv')
data_1inor_1org_1edge.to_csv('combined_data_frame_1inorg_1org_1edge.csv')
data_2inor_1edge.to_csv('combined_data_frame_2inorg_1edge.csv')

