import pandas as pd


# CSV file name for untrained agent
csv_file2 = "../logs/eval_without_model.monitor.csv"

df = pd.read_csv(csv_file2, skiprows=1)

mean_reward = df['r'].mean()
mean_length = df['l'].mean()
mean_times = df['t'].mean()
mean_distance_covered = df['distance_covered'].mean()
success_rate = len(df[df['l'] == 40]) / len(df) * 100

print("Evaluation for untrained agent:")
print("Mean reward:", mean_reward)
print("Mean episode length (time in s):", mean_length)
print("Mean runtime in seconds of all the episodes:", mean_times)
print("Mean distance covered:", mean_distance_covered)
print(f"Percentage of success rate: {success_rate}% (when vehicle did not crash during an episode; {100 - success_rate}% of times it crashed)")
print()

# CSV file name for trained agent
csv_file = "../logs/eval_with_PPO_model.monitor.csv"

df2 = pd.read_csv(csv_file, skiprows=1)

mean_reward2 = df2['r'].mean()
mean_length2 = df2['l'].mean()
mean_times2 = df2['t'].mean()
mean_distance_covered2 = df2['distance_covered'].mean()
success_rate2 = len(df2[df2['l'] == 40]) / len(df2) * 100


print("Evaluation for trained agent:")
print("Mean reward:", mean_reward2)
print("Mean episode length (time in s):", mean_length2)
print("Mean runtime in seconds of all the episodes:", mean_times2)
print("Mean distance covered:", mean_distance_covered2)
print(f"Percentage of success rate: {success_rate2}% (when the vehicle did not crash during an episode; {100 - success_rate2}% of times it crashed)")
print()

# Percentage improvements
per_imp_mean_reward = (mean_reward2 - mean_reward) / mean_reward * 100
per_imp_mean_length = (mean_length2 - mean_length) / mean_length * 100
per_imp_mean_distance_covered = (mean_distance_covered2 - mean_distance_covered) / mean_distance_covered * 100

print(f"Percentage improvement based on mean reward: {per_imp_mean_reward}%")
print(f"Percentage improvement based on mean episode length (time in s): {per_imp_mean_length}%")
print(f"Percentage improvement based on mean distance covered: {per_imp_mean_distance_covered}%")
