import numpy as np
import pandas as pd

np.random.seed(42)

song_data = pd.read_csv('data/song_data.csv')

# Restrict unpopular data to that which is less or equal to 25 out of 100
unpopular_songs = song_data[song_data['Is_Popular'] == 0]

# Unpopular data greatly outnumbers popular data, which makes KNN less accurate
# Shuffles unpopular data and selects a subset approximately equal to the set of popular data
shuffling = np.random.permutation(len(unpopular_songs))
shuffled_songs = unpopular_songs.iloc[shuffling].reset_index(drop=True)
num_rows = len(unpopular_songs) // 7
unpopular_songs = shuffled_songs.head(num_rows)

popular_songs = song_data[song_data['Is_Popular'] == 1]

# Recombines popular and unpopular data
song_data = pd.concat([unpopular_songs, popular_songs])

song_data.to_csv('data/selected_data.csv', index=False)