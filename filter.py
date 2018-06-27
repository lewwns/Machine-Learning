import csv
from musixmatch import Musixmatch
musixmatch = Musixmatch('d9f4b70591c4995eaa6ae7a0c508d465')


def check_language(mmID):
    try:
        lyrics = musixmatch.track_lyrics_get(mmID)
        language = lyrics['message']['body']['lyrics']['lyrics_language_description']
    except:
        language = 'haha'

    return language


all_lan_dataset = open("Country_Rap_12000_2.csv").readlines()
#print len(all_lan_dataset)
en_song_list = []
for i in range(0, len(all_lan_dataset)):
    #print i
    ID_list = []
    song = all_lan_dataset[i]
    info = song.strip().split(",")
    #print info
    mmID = info[3]
    ID_list.append(mmID)
    
    if check_language(mmID) == 'English' or check_language(mmID) =='en':
        en_song_list.append(ID_list)
    


en_song_output = open('en_song_country_rap.csv','w')
with en_song_output:
    writer = csv.writer(en_song_output)
    writer.writerows(en_song_list)


