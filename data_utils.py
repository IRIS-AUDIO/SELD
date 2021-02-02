import torchaudio

def load_wav(path):
    from glob import glob
    return list(map(torchaudio.load, glob('*.wav')))