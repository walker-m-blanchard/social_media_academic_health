import re
import csv
import contractions
from nltk.tokenize import word_tokenize as tokenize
from nltk.corpus import stopwords as stop
import pandas as pd
import string

# Individual songs bank of words
class WordBank:
    def __init__(self, words, combined_bank):
        self.bank = {}
        for word in words:
            if word not in combined_bank:
                self.bank[word] = 1
                combined_bank[word] = 1
            elif word not in self.bank:
                self.bank[word] = 1
                combined_bank[word] += 1
            else:
                self.bank[word] += 1
                combined_bank[word] += 1

def preprocessing(message:str):
    # Removes leading and trailing whitespace
    message = message.strip()
    message = message.lower()
    message = " ".join(message.split())
    # Removes brackets and their contents from beginning of lyrics
    message = re.sub(r"\[.*?]", "", message)

    # Removes non-english characters
    message = message.encode("ascii", "ignore").decode('ascii')

    # Separates contractions into separate words
    message = contractions.fix(message)

    words = tokenize(message)
    # Removes stopwords
    words = [word for word in words if word not in stop_words]
    # Removes punctuation
    words = [word for word in words if word not in string.punctuation]

    ## Don't know if this next section is necessary anymore
    ## Was used before separating contractions
    # Removes leading and trailing punctuation
    for i in range(len(words)):
        if words[i][0] in string.punctuation:
            words[i] = words[i][1:]
        if words[i][-1] in string.punctuation:
            words[i] = words[i][:-1]

    # Removes leftovers from previous stop-word removal
    for word in words:
        if len(word) < 2:
            words.remove(word)
    ## End uncertain section

    return words

# Stopwords identified in NLTK database
stop_words = stop.words('english')

lyric_data = pd.read_csv("data/hf_data.csv", usecols=['lyrics'], dtype=str)

songs_bank = []
# Bank of words for all songs combined
total_bank = {}
for song in lyric_data.itertuples():
    tokens = preprocessing(song[1])
    new_song = WordBank(tokens, total_bank)
    songs_bank.append(new_song)

header = ['word', 'count']
with open('data/total_bank.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for key, value in total_bank.items():
        writer.writerow([key, value])