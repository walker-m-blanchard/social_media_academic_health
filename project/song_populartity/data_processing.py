import pandas as pd

song_data = pd.read_csv('data/hf_data.csv')

# Convert non-numerical data to numerical
song_data['popularity'] = pd.to_numeric(song_data['popularity'], errors='coerce')
song_data['explicit'] = pd.to_numeric(song_data['explicit'], errors='coerce')

# Assign data as popular if popularity is greater than or equal to 75 out of 100
song_data['Is_Popular'] = (song_data['popularity'] >= 75).astype(int)
# Restrict unpopular data to that which is less or equal to 25 out of 100
unpopular_songs = song_data[song_data['popularity'] <= 25]
# Remove 0 popularity data due to noise
unpopular_songs = unpopular_songs[unpopular_songs['popularity'] > 0]

popular_songs = song_data[song_data['popularity'] >= 75]

# Recombines popular and unpopular data
song_data = pd.concat([unpopular_songs, popular_songs])

# Selects attributes which are of interest
song_data = song_data[['danceability', 'energy', 'key', 'mode', 'speechiness', 'instrumentalness', 'liveness', 'tempo',
                      'loudness', 'acousticness', 'valence', 'duration_ms', 'Is_Popular', 'track_genre', 'lyrics']]

song_data.to_csv('data/song_data.csv', index=False)