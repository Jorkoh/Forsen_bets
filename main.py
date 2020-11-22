import requests
import json
from collections import Counter

# True to scrap new data from web
DATA_FROM_WEB = False


def main():
    data = get_data()
    print_most_common_top_predictors(data)


def print_most_common_top_predictors(data):
    top_predictors_names = []
    top_predictors_names += [predictor['user_display_name']
                             for bet in data['data'].values()
                             for outcome in bet['event']['outcomes']
                             for predictor in outcome['top_predictors']]

    counter = Counter(top_predictors_names)
    print(f"{len(counter.keys())} different top predictors found")
    print('\n' + '\n'.join([f"{predictor[0]} appears {predictor[1]} times" for predictor in counter.most_common(20)]))


def get_data():
    if DATA_FROM_WEB:
        data_id = requests.get("https://api.twitchpredict.com/api/v1/data/id/forsen").json()['data']
        data = requests.get(f"https://api.twitchpredict.com/api/v1/data/events/{data_id}").json()
        with open('data.json', 'w') as data_file:
            json.dump(data, data_file)
    else:
        with open('data.json') as data_file:
            data = json.load(data_file)
    return data


if __name__ == "__main__":
    main()
