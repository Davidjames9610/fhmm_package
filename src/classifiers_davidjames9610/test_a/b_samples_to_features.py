import numpy as np
import random
import src.ads_davidjames9610.useful as useful
import matplotlib.pyplot as plt
import src.misc_davidjames9610.noisey as noisey
import src.classifiers_davidjames9610.test_a.config as base_config
import importlib

config = importlib.import_module(base_config.config_location)


def get_average_power_for_samples(cv_output):
    # get average signal power
    curr_samples = cv_output['train_data'][0]
    random_indices = random.sample(range(1, len(curr_samples)), 20)
    samples = np.concatenate(curr_samples[:20])  # np.concatenate([curr_samples[indice] for indice in random_indices])
    ap = periodic_power(samples, 500, 250)
    ap = np.array(ap)
    # ap = ap[ap > 0.05]
    plt.plot(ap)
    plt.show()
    plt.close()
    return np.mean(ap)


def periodic_power(x, lx, p):
    buf = buffer(x, lx, p)
    average_pow = []
    for b in buf:
        average_pow.append(np.sum(np.square(b)) / lx)
    return average_pow


def buffer(x, n, p=0, opt=None):
    """
    Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from x
    """
    import numpy as np

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(x):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = x[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), x[:n - p]])
                i = n - p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = x[i:i + (n - p)]
        if p != 0:
            col = np.hstack([result[:, -1][-p:], col])
        i += n - p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result.T


def get_real_noise_sample(noise_key, target_snr_db, signal_db, sample_len, sr):
    # 1. read in noise sample and normalise
    noise_sample = useful.file_to_audio(noise_key, sr)[0]
    noise_sample = noise_sample[100000:]  # update to avoid silence at start
    ap = periodic_power(noise_sample, 200, 100)
    ap = np.array(ap)
    # ap = ap[ap > 0.05]
    # plt.plot(ap)
    # plt.title('periodic power noise')
    # plt.show()
    # plt.close()

    if len(noise_sample) < sample_len:
        print('oh no; len(noise_sample) < sample_len')

    noise_avg_watts_v2 = np.square(np.linalg.norm(noise_sample, ord=2)) / len(
        noise_sample)

    # 2. Calculate the scaling factor
    noise_avg_watts = np.mean(noise_sample ** 2)

    noise_target_db = signal_db - target_snr_db
    noise_target_watts = 10 ** (noise_target_db / 10)

    scaling_factor = np.sqrt(noise_target_watts / noise_avg_watts)

    # 3. return noise scaled
    scaled_noise_sample = noise_sample * scaling_factor
    # 10 * np.log10(np.mean(scaled_noise_sample ** 2))
    return scaled_noise_sample


from src.misc_davidjames9610.noisey import get_noise_avg_watts


# import importlib
# importlib.reload(fe)

class ProcessingBase:
    def __init__(self, fe_method):
        self.fe_method = fe_method

    def __str__(self):
        return self.fe_method.__str__() + '_None_None'

    def pre_process(self, sample):
        return self.fe_method(sample)

    def post_process(self, sample):
        return self.fe_method(sample)

    def get_noise_feature(self):
        pass


class ProcessingGaussNoise:
    def __init__(self, fe_method, snr, signal_power, sample_len):
        self.fe_method = fe_method
        self.snr = snr
        self.signal_power = signal_power
        self.noise_sample = noisey.get_noise_for_sample(sample_len, self.signal_power, self.snr, self.snr, self.snr)

    def __str__(self):
        return self.fe_method.__str__() + '_GaussNoise_SNR' + str(self.snr)

    def pre_process(self, sample):
        return self.fe_method(sample)

    def post_process(self, sample):
        sample = self.noise_sample[:len(sample)] + sample
        return self.fe_method(sample)

    def get_noise_feature(self):
        return self.fe_method(self.noise_sample)


class ProcessingRealNoise:
    def __init__(self, fe_method, snr, signal_power, sample_len, noise_key):
        self.fe_method = fe_method
        self.snr = snr
        self.signal_power = signal_power
        self.noise_key = noise_key

        self.noise_sample = get_real_noise_sample(
            noise_key=base_config.noise_sound_lib[noise_key], target_snr_db=snr, signal_db=signal_power,
            sample_len=sample_len, sr=config.sr)

        print('completed ProcessingRealNoise')

    def __str__(self):
        return self.fe_method.__str__() + '_RealNoise_' + self.noise_key + '_SNR' + str(self.snr)

    def pre_process(self, sample):
        return self.fe_method(sample)

    def post_process(self, sample):
        sample = self.noise_sample[:len(sample)] + sample
        return self.fe_method(sample)

    def get_noise_feature(self):
        return self.fe_method(self.noise_sample)


class ProcessingGaussNoiseReverb:
    # todo finish this somehow
    def __init__(self, fe_method, snr, signal_power, sample_len):
        self.snr = snr
        self.signal_power = signal_power
        self.fe_method = fe_method

    def __str__(self):
        return self.fe_method.__str__() + '_GaussNoise_SNR' + str(self.snr) + '_Reverb'

    def pre_process(self, sample):
        return self.fe_method(sample)

    def post_process(self, sample):
        # todo update so Gauss Noise Reverb works
        return self.fe_method(sample)

    def get_noise_feature(self):
        # update
        return []

# snr = signal - noise
