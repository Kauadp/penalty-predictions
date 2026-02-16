from yt_dlp import YoutubeDL

PLAYLIST_URL = "https://www.youtube.com/watch?v=MCWJNOfJoSM&list=PLCGIzmTE4d0hAPk0xOuuTaOyK2Qxr0atu&pp=0gcJCbUEOCosWNin"

ydl_opts = {
    "format": "mp4[height<=720]/best[height<=720]",
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([PLAYLIST_URL])
