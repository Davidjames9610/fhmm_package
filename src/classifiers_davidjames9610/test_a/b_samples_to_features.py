import numpy as np
import random
import src.ads_davidjames9610.useful as useful

def get_average_power_for_samples(cv_output):
    # get average signal power
    curr_samples = cv_output['train_data'][0]
    random_indices = random.sample(range(1, len(curr_samples)), 20)
    samples = np.concatenate([curr_samples[indice] for indice in random_indices])
    ap = periodic_power(samples, 500, 250)
    ap = np.array(ap)
    ap = ap[ap > 0.05]
    # plt.plot(ap)
    # plt.show()
    # plt.close()
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

    if len(noise_sample) < sample_len:
        print('oh no; len(noise_sample) < sample_len')

    # 2. Calculate the scaling factor
    noise_avg_watts = np.mean(noise_sample ** 2)

    noise_target_db = signal_db - target_snr_db
    noise_target_watts = 10 ** (noise_target_db / 10)

    scaling_factor = np.sqrt(noise_target_watts / noise_avg_watts)

    # 3. return noise scaled
    scaled_noise_sample = noise_sample * scaling_factor

    return scaled_noise_sample
