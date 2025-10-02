import pandas as pd

lyrics_data = pd.read_csv("data/labeled_lyrics_cleaned.csv", usecols=['artist', 'song', 'seq'])
num_data = pd.read_csv("data/spotify_songs.csv")

num_data = num_data.drop(columns=['track_id', 'track_album_id', 'track_album_name', 'track_album_release_date',
                                  'playlist_name', 'playlist_id'])


song_data = pd.merge(lyrics_data, num_data, left_on=['artist', 'song'], right_on=['track_artist', 'track_name'], how='inner')
song_data = song_data.drop(columns=['artist', 'song'])
song_data.drop_duplicates(subset=['track_name', 'track_artist'], inplace=True)

song_data.to_csv("data/kaggle_data.csv", index=False)