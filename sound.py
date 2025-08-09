import numpy as np
import wave
import struct

def generate_beep(filename='alert.wav', duration=0.5, freq=440, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))
    data = 0.5 * np.sin(2 * np.pi * freq * t)  # sine wave at freq Hz

    with wave.open(filename, 'w') as wav_file:
        nchannels = 1
        sampwidth = 2  # bytes per sample
        framerate = sample_rate
        nframes = len(data)
        comptype = "NONE"
        compname = "not compressed"
        wav_file.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))

        for s in data:
            wav_file.writeframes(struct.pack('h', int(s * 32767)))

generate_beep()
print("alert.wav generated!")
