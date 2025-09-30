import pandas as pd
import numpy as np

lyrics_data = pd.read_csv("data/spotify_millsongdata.csv", usecols=['artist', 'song', 'text'])
num_data = pd.read_csv("data/spotify_songs.csv")

num_data = num_data.drop(columns=['track_id', 'track_album_id', 'track_album_name', 'track_album_release_date',
                                  'playlist_name', 'playlist_id'])


song_data = pd.merge(lyrics_data, num_data, left_on=['artist', 'song'], right_on=['track_artist', 'track_name'], how='inner')
song_data = song_data.drop(columns=['artist', 'song'])

song_data.to_csv("data/song_data.csv", index=False)