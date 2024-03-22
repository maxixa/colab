# youtube

play = ""

from pytube import Playlist
from pytube import YouTube



playlist = Playlist(play)
print('Number of videos in playlist: %s' % len(playlist.video_urls))
for video_url in playlist.video_urls:
    print(video_url)
    video=YouTube(video_url)
    try:
       #video.streams.first().download()
       video.streams.filter(res="480p").first().download()
       #video.streams.filter(res="720p").first().download()

    except:
         continue

from pytube import Playlist
from pytube import YouTube
playlist = Playlist(play)
print('Number of videos in playlist: %s' % len(playlist.video_urls))
for video_url in playlist.video_urls:
    print(video_url)
    video=YouTube(video_url)
    try:
       video.streams.get_by_itag(249).download()
    except:
         continue