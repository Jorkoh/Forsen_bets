import requests
import json
from collections import Counter

# True to scrap new data from web
DATA_FROM_WEB = False

if DATA_FROM_WEB:
    data_id = requests.get("https://api.twitchpredict.com/api/v1/data/id/forsen").json()['data']
    data = requests.get(f"https://api.twitchpredict.com/api/v1/data/events/{data_id}").json()
    with open('data.json', 'w') as data_file:
        json.dump(data, data_file)
else:
    with open('data.json') as data_file:
        data = json.load(data_file)

top_predictors_names = []
for bet_id, bet in data['data'].items():
    for outcomes in bet['event']['outcomes']:
        top_predictors_names += [predictor['user_display_name'] for predictor in outcomes['top_predictors']]

c = Counter(top_predictors_names)
print(len(c.keys()))
print(c.most_common(30))
print([x[0] for x in c.most_common(30)])

# text = json.dumps(data.json(), sort_keys=True, indent=4)
# print(text.count('\n'))