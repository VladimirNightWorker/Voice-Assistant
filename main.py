import os
import torch
from gtts import gTTS
from io import BytesIO
import sounddevice as sd
import soundfile as sf


def speaker_gtts(text):
    lang = os.getenv('LANG')
    with BytesIO() as f:
        gTTS(text=text, lang=lang, slow=False).write_to_fp(f)
        f.seek(0)
        data, fs = sf.read(f)
        sd.play(data, fs, blocking=True)


models_urls = ['https://models.silero.ai/models/tts/ru/v3_1_ru.pt', 'https://models.silero.ai/models/tts/en/v3_en.pt']

model_ru = 'silero_models/ru/model.pt'
model_en = 'silero_models/en/model.pt'

device = torch.device('cpu')
torch.set_num_threads(6)   # 4
local_file = 'model_ru'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(models_urls[0], local_file)   # 'https://models.silero.ai/models/tts/ru/v3_1_ru.pt'


model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

sample_rate = 48000
speaker = 'baya'  # kseniya, aidar, baya, xenia, eugene, random
en_speaker = 'en_6'   # от 0 -117


audio_paths = model.save_wav(text='Владик бес',
                             speaker=speaker,
                             sample_rate=sample_rate)


def speaker_silero(text):
    audio = model.apply_tts(text=text,
                            speaker=speaker,
                            sample_rate=sample_rate)

    sd.play(audio, blocking=True)


if __name__ == '__main__':
    # speaker_silero(text=input('\nНапиши сюда...\n\n'))
    speaker_silero(text='Ириша иди')
