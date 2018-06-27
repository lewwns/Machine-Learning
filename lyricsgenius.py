import lyricwikia
import csv

song_data = []
with open("en_jazz_rock_2699.csv", newline='',encoding='utf-8') as song_dataset:
    reader = csv.reader(song_dataset, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

    for row in reader:
        song_data.append(row)


song_list = []
for i in range(1, len(song_data)):
    song = song_data[i]
    song_info = []
    #info = song.strip().split(",")
    
    title = song[2]
    artist = song[1]
    genre = song[0]

    try:
        lyrics = lyricwikia.get_lyrics(artist, title)

        song_info.append(title)
        song_info.append(artist)
        song_info.append(genre)
        song_info.append(lyrics)

        #print(song_info)

        song_list.append(song_info)

    except:
        continue

print('start')
'''
i=1
output = "song_output.csv"
outputfile = open(output, 'a')
for song in song_list:
    print(i)
    outputfile.write('{}, {}, {}, {}\n'
                   .format(song[0], song[1], song[2], song[3]))
    i+=1

outputfile.close()
'''

print(len(song_list))

song_output = open('song_rock_jazz.csv', 'w')
with song_output:
    writer = csv.writer(song_output)
    for row in song_list:
        try:
            writer.writerow(row)
        #writer.writerows(song_list)

        except UnicodeEncodeError:
            continue



