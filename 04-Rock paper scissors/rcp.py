
import random


choices = ["Rock", "Paper", "Scissors"]


player_strategies = {

    1: lambda: "Rock",  


    2: lambda: random.choice(choices),  


    
    3: lambda: random.choice(["Rock", "Paper"])
}


def decide_winner(player_move, ai_move):
    if player_move == ai_move:
        return "draw"
    elif (player_move == "Rock" and ai_move == "Scissors") or \
         (player_move == "Paper" and ai_move == "Rock") or \
         (player_move == "Scissors" and ai_move == "Paper"):
        return "player"
    else:
        return "ai"

# Simulate multiple games for a given mode


def run_simulation(mode, rounds=100):
    wins = 0
    pick_move = player_strategies[mode]

    for _ in range(rounds):
        user_move = pick_move()
        bot_move = random.choice(choices)
        outcome = decide_winner(user_move, bot_move)

        if outcome == "player":
            wins += 1

    return wins


# Run and print results for each mode
for mode in range(1, 4):
    player_wins = run_simulation(mode)
    win_ratio = player_wins / 100
    print(f"Mode {mode}:")
    print(f"Player win rate: {win_ratio:.2f}")
    print()
