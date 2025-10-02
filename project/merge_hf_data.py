import pandas as pd

lyrics_data = pd.read_csv("data/song_lyrics2.csv", usecols=['artist', 'title', 'lyrics'])
num_data = pd.read_csv("data/dataset.csv")

num_data = num_data.drop(columns=['track_id', 'album_name', 'Unnamed: 0'])

song_data = pd.merge(lyrics_data, num_data, left_on=['artist', 'title'], right_on=['artists', 'track_name'], how='inner')
song_data = song_data.drop(columns=['artists', 'track_name'])
song_data.drop_duplicates(subset=['artist', 'title'], inplace=True)

song_data.to_csv("data/hf_data.csv", index=False)