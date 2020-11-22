import requests
import json
from collections import Counter
import matplotlib.pyplot as plt

# True to scrap new data from web
DATA_FROM_WEB = False


def main():
    bets = get_bets()
    bets = clean_and_sort_bets(bets)

    # print_most_common_top_predictors(bets)
    # print_bet_results(bets)
    print_outcome_pov(bets, True)
    print_outcome_pov(bets, False)


def clean_and_sort_bets(bets):
    valid_bets = list(filter(lambda bet:
                             bet['event']['title'] != "Fake prediction answer is blue."
                             and bet['event']['status'] == "RESOLVED", bets))
    valid_bets.sort(key=lambda b: b['timestamp'])
    return valid_bets


def print_bet_results(bets):
    winner_color_counts = {}
    for bet in bets:
        winner = 'UNKNOWN'
        for outcome in bet['event']['outcomes']:
            if outcome['id'] == bet['event']['winning_outcome_id']:
                winner = outcome['color']
        if winner not in winner_color_counts:
            winner_color_counts[winner] = 1
        else:
            winner_color_counts[winner] += 1
    print(winner_color_counts)


def print_outcome_pov(bets, believer):
    # get rid of the sniper game bets since they were kinda scuffed
    bets = list(filter(lambda bet: bet['event']['title'] != "Which team will win?", bets))

    if believer:
        color_choice = "BLUE"
    else:
        color_choice = "PINK"

    points = 100_000
    betting_percentage = 0.15

    points_track = [points]

    for bet in bets:
        stake = min(round(points * betting_percentage), 250_000)
        print(f"Question: {bet['event']['title']}")
        print(f"Stake: {stake}")

        # Find the winner and the odds
        winner_color = 'UNKNOWN'
        winner_odds = 0
        for outcome in bet['event']['outcomes']:
            if outcome['id'] == bet['event']['winning_outcome_id']:
                winner_color = outcome['color']
                other_outcomes_points = sum(outcome['total_points'] for outcome in bet['event']['outcomes']
                                            if outcome['color'] != winner_color)
                winner_odds = other_outcomes_points / outcome['total_points']
                print(f"Decimal odds for winner: {round(winner_odds + 1, 2)}")
        if winner_color == 'UNKNOWN':
            raise ValueError("Couldn't find the winner among the outcomes")

        # Assign the payout
        if color_choice == winner_color:
            print(f"Won {round(stake * winner_odds)} points!")
            points += round(stake * winner_odds)
        else:
            print(f"Lost {stake} points!")
            points -= stake
        print(f"Remaining points: {points}\n")
        points_track.append(points)

    fig, ax = plt.subplots()
    ax.set_ylabel('Points')
    ax.set_xlabel('Bets')
    if believer:
        ax.set_title("BELIEVER POV")
        color = "blue"
    else:
        ax.set_title("DOUBTER POV")
        color = "red"
    ax.plot(points_track, color=color)
    fig.tight_layout()
    plt.show()


def print_most_common_top_predictors(bets):
    top_predictors_names = []
    top_predictors_names += [predictor['user_display_name']
                             for bet in bets
                             for outcome in bet['event']['outcomes']
                             for predictor in outcome['top_predictors']]

    counter = Counter(top_predictors_names)

    print(f"{len(counter.keys())} different top predictors found")
    common = counter.most_common(20)
    print('\n' + '\n'.join([f"{predictor[0]} appears {predictor[1]} times" for predictor in common]))

    fig, ax = plt.subplots()
    ax.bar([predictor[0] for predictor in counter.most_common(20)], [predictor[1] for predictor in common])
    ax.tick_params(axis='x', which='major', labelsize=7)
    fig.autofmt_xdate()
    plt.show()


def get_bets():
    if DATA_FROM_WEB:
        data_id = requests.get("https://api.twitchpredict.com/api/v1/data/id/forsen").json()['data']
        data = requests.get(f"https://api.twitchpredict.com/api/v1/data/events/{data_id}").json()
        with open('data.json', 'w') as data_file:
            json.dump(data, data_file)
    else:
        with open('data.json') as data_file:
            data = json.load(data_file)
    return list(data['data'].values())


if __name__ == "__main__":
    main()
