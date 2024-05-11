import subprocess
import os
from youtube_transcript_api import YouTubeTranscriptApi
from huggingsound import SpeechRecognitionModel
from pytube import YouTube
import soundfile as sf
import librosa
import torch
import re

def extract_video_id(url):
    pattern = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def setup(url):
    yt = YouTube(url)
    audio_file = 'data/ytaudio.mp4'
    wav_file = 'data/ytaudio.wav'

    yt.streams.filter(only_audio=True, file_extension='mp4').first().download(filename=audio_file)

    # Convert mp4 to wav using FFmpeg
    subprocess.run(['ffmpeg', '-i', audio_file, '-acodec', 'pcm_s16le', '-ar', '16000', wav_file])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)
    return model

def extract_text_without_subtitles(url):
    model = setup(url)
    input_file = 'data/ytaudio.wav'
    stream = librosa.stream(
        input_file,
        block_length=30,
        frame_length=16000,
        hop_length=16000
    )
    for i, speech in enumerate(stream):
        sf.write(f'data/{i}.wav', speech, 16000)

    audio_paths = [f'data/{a}.wav' for a in range(i + 1)]
    transcriptions = model.transcribe(audio_paths)

    full_transcript = ''
    for item in transcriptions:
        full_transcript += ' '.join(item['transcription'])

    return full_transcript.strip()


def delete_audio_files():
    for file in os.listdir('data'):
        if file.endswith('.wav') or file.endswith('.mp4'):
            os.remove(os.path.join('data', file))

def choose_extract(transcript):
    if transcript==True:
        extracted_text = ''.join([d['text'] for d in transcript])  #with subtitles
    else:
        extracted_text = extract_text_without_subtitles(video_url)  #without subtitles
        delete_audio_files()
    return extracted_text


video_url = input("Enter video url: ")
transcript = YouTubeTranscriptApi.get_transcript(extract_video_id(video_url))
extracted_text=choose_extract(transcript)
print(extracted_text)