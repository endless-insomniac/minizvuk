import torchaudio
import torch
import math


class WaveformToSpectrogram:
    def __init__(self, n_fft=1724, win_length=1724, hop_length=130,
                 target_frames=600, random_crop=False):
        self.target_frames = target_frames
        self.random_crop = random_crop

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                                             window_fn=torch.blackman_window, power=2.0)

    def __call__(self, waveform):
        spectrogram = self.spectrogram(waveform)
        spectrogram = torch.log(spectrogram + 1e-8)
        spectrogram = self.fix_num_frames(spectrogram)
        return spectrogram

    def fix_num_frames(self, spectrogram):
        num_frames = spectrogram.shape[-1]
        target = self.target_frames

        if num_frames > target:
            if self.random_crop:
                start = torch.randint(0, num_frames -target + 1, (1,)).item()
            else:
                start = 0

            return spectrogram[..., start:start + target]

        elif num_frames == target:
            return spectrogram

        else:
            repeat_rate = math.ceil(target / num_frames)
            repeat_shape = [1] * spectrogram.ndim
            repeat_shape[-1] = repeat_rate

            return spectrogram.repeat(*repeat_shape)[..., :target]

