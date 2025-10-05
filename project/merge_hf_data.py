import pandas as pd

lyrics_data = pd.read_csv("data/song_lyrics2.csv", usecols=['artist', 'title', 'lyrics', 'language'], header=0)
num_data = pd.read_csv("data/dataset.csv")

# Drops column used to number the songs from 1...
num_data = num_data.drop(columns=['album_name', 'Unnamed: 0'])

# Merges data by artist and song, drops the duplicate columns, drops duplicate songs, and filters for English
song_data = pd.merge(lyrics_data, num_data, left_on=['artist', 'title'], right_on=['artists', 'track_name'], how='inner')
song_data.drop(columns=['artists', 'track_name'], inplace=True)
song_data.drop_duplicates(subset=['artist', 'title'], inplace=True)
filtered_songs = song_data[song_data['language'] == 'en']

filtered_songs.to_csv("data/hf_data.csv", index=False)