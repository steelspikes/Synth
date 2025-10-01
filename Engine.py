import sounddevice as sd

class Engine:
    def __init__(self, sample_rate=44100):
        self.modules = []
        self.sample_rate = sample_rate

    def play(self, sample):
        sd.play(sample, self.sample_rate)