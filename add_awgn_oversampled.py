import numpy as np

def add_awgn_oversampled(signal, ebno_db, samples_per_symbol=8, bits_per_symbol=4):
    """
    Adds complex AWGN to an oversampled signal based on target Eb/N0.
    """
    # 1. Convert Eb/N0 to Es/N0 (Energy per Symbol)
    esno_db = ebno_db + 10 * np.log10(bits_per_symbol)
    esno_linear = 10 ** (esno_db / 10)
    
    # 2. Measure the average power of your oversampled signal
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # 3. Calculate required noise variance per sample, accounting for oversampling
    # Total noise power in oversampled bandwidth = (Signal Power / Es/N0) * SPS
    noise_power = (signal_power / esno_linear) * samples_per_symbol
    
    # 4. Generate complex Gaussian noise (divide power by 2 for I and Q channels)
    noise_std = np.sqrt(noise_power / 2)
    noise_i = np.random.normal(0, noise_std, signal.shape)
    noise_q = np.random.normal(0, noise_std, signal.shape)
    noise = noise_i + 1j * noise_q
    
    # 5. Return the noisy signal
    return signal + noise

# --- Usage Example ---
# rx_signal_noisy = add_awgn_oversampled(tx_signal_out, ebno_db=15)
