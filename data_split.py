
import json
import os

jsons = os.listdir("./TickJson")
json_list = []
for js in jsons:
    if ".json" in js: json_list.append(js)

N_json = len(json_list)
split_ratio = 9/10
for ijson in range(int(split_ratio * N_json)):
    os.rename("./TickJson/" + json_list[ijson], "./TickJson/train/" + json_list[ijson])
for ijson in range(int(split_ratio * N_json), N_json):
    os.rename("./TickJson/" + json_list[ijson], "./TickJson/test/" + json_list[ijson])