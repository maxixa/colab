import os
import subprocess

video_dir = ""

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        
        base_name = os.path.splitext(video_file)[0]
        audio_file = base_name + '.webm'
        
        output_file = base_name + '-merged.mp4'
        
        ffmpeg_command = [
            'ffmpeg', 
            '-i', os.path.join(video_dir, video_file),
            '-i', os.path.join(video_dir, audio_file),
            '-c:v', 'copy',
            '-c:a', 'copy',
            os.path.join(video_dir, output_file)
        ]
        
        subprocess.run(ffmpeg_command)