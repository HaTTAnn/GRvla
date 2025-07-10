import pandas as pd
import numpy as np
import json
# df = pd.read_parquet("/data/dataset/real/aloha/data/clean_data_without_split/lerobot/pick_put_banana_0604/data/chunk-000/episode_000009.parquet")
# df = pd.read_parquet("/data/dataset/real/aloha/data/clean_data_without_split/lerobot/pick_put_banana_0616/data/chunk-000/episode_000009.parquet")
# print(df)
# def clean_column(df, col_name):
#     for i in range(len(df)):
#         row = df.at[i, col_name]

#         if isinstance(row, np.ndarray):
#             row = row.tolist()
#         elif not isinstance(row, list):
#             continue  

#         modified = False

#         if len(row) > 6 and row[6] < 0:
#             row[6] = 0
#             modified = True
#         if len(row) > 13 and row[13] < 0:
#             row[13] = 0
#             modified = True

#         if modified:
#             df.at[i, col_name] = row  

# clean_column(df,"action")
# clean_column(df,"observation.state")
        
# for i in range(len(df)):
#     state = df.at[i, "observation.state"]

#     if isinstance(state, np.ndarray):
#         state = state.tolist()
#     elif not isinstance(state, list):
#         continue

#     if len(state) > 13:
#         if state[6] >= 0.002 or state[13] >= 0.002:
#             print(f"Row {i}: state[6]={state[6]:.3f}, state[13]={state[13]:.3f}, timestamp={df.at[i, 'timestamp']}")

# for i in range(len(df)):
#     state = df.at[i, "action"]

#     if isinstance(state, np.ndarray):
#         state = state.tolist()
#     elif not isinstance(state, list):
#         continue

#     if len(state) > 13:
#         if state[6] <0 or state[13] <0:
#             print(f"Row {i}: action[6]={state[6]:.3f}, action[13]={state[13]:.3f}, timestamp={df.at[i, 'timestamp']}")



with open("/data/dataset/real/aloha/data/clean_data_without_split/lerobot/pick_put_banana_0604/meta/episodes_stats.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        for i in range(14):
            max = record["stats"]["action"]["max"][i]
            min = record["stats"]["action"]["max"][i]
    



