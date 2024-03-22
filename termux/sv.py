link = ""

from pytube import YouTube
video_url = link
video=YouTube(video_url)
video.streams.filter(res="480p").first().download()
# video.streams.filter(res="720p",adaptive=True).first().download()

from pytube import YouTube
video_url = link
video=YouTube(video_url)
video.streams.get_by_itag(249).download()
# video.streams.get_by_itag(140).download()