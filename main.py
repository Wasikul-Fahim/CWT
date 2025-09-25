import numpy as np
import matplotlib.pyplot as plt
import pywt  # PyWavelets - the standard wavelet library
from scipy.stats import kurtosis
import seaborn as sns

class CWTKurtogramAnalyzer:
    def __init__(self, fs=1000):
        """
        Initialize the CWT and Kurtogram analyzer using PyWavelets
        
        Parameters:
        fs (int): Sampling frequency
        """
        self.fs = fs
        self.time = None
        self.signal = None
        self.cwt_matrix = None
        self.frequencies = None
        self.scales = None
        
    def generate_sample_signals(self, signal_type='chirp', duration=2, noise_level=0.1):
        """Generate various sample signals for testing"""
        from scipy import signal as scipy_signal
        
        t = np.linspace(0, duration, int(duration * self.fs), endpoint=False)
        self.time = t
        
        if signal_type == 'chirp':
            self.signal = scipy_signal.chirp(t, f0=10, f1=100, t1=duration, method='linear')
            
        elif signal_type == 'multi_sine':
            self.signal = (np.sin(2*np.pi*10*t) + 
                          0.5*np.sin(2*np.pi*25*t) + 
                          0.3*np.sin(2*np.pi*50*t))
            
        elif signal_type == 'impulse':
            impulse_times = [0.5, 1.0, 1.5]
            self.signal = np.zeros_like(t)
            for imp_time in impulse_times:
                idx = int(imp_time * self.fs)
                if idx < len(self.signal):
                    decay = np.exp(-10*(t[idx:] - imp_time))
                    oscillation = np.sin(2*np.pi*30*(t[idx:] - imp_time))
                    self.signal[idx:] += decay * oscillation
                    
        elif signal_type == 'modulated':
            carrier = 50
            modulator = 5
            self.signal = (1 + 0.5*np.sin(2*np.pi*modulator*t)) * np.sin(2*np.pi*carrier*t)
            
        elif signal_type == 'bearing_fault':
            fault_freq = 10
            resonance_freq = 300  # Reduced for better visualization
            self.signal = np.random.normal(0, 0.1, len(t))
            
            for i in range(int(duration * fault_freq)):
                imp_time = i / fault_freq
                if imp_time < duration:
                    idx = int(imp_time * self.fs)
                    if idx < len(self.signal):
                        decay_length = int(0.05 * self.fs)
                        end_idx = min(idx + decay_length, len(self.signal))
                        decay = np.exp(-50*(t[idx:end_idx] - imp_time))
                        oscillation = np.sin(2*np.pi*resonance_freq*(t[idx:end_idx] - imp_time))
                        self.signal[idx:end_idx] += 2 * decay * oscillation
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(self.signal))
            self.signal += noise
            
        return self.time, self.signal
    
    def compute_cwt(self, wavelet='cmor1.5-1.0', scales=None, num_scales=50):
        """
        Compute CWT using PyWavelets
        
        Parameters:
        wavelet (str): Wavelet name ('cmor1.5-1.0', 'morl', 'mexh', 'gaus1')
        scales (array): Custom scales
        num_scales (int): Number of scales if not provided
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Generate or load a signal first.")
        
        if scales is None:
            # Create scales for desired frequency range
            freq_min = 1
            freq_max = self.fs // 4  # Nyquist frequency / 2 for safety
            
            # For complex Morlet wavelet, use frequency-to-scale conversion
            if 'cmor' in wavelet or 'morl' in wavelet:
                # Get central frequency of the wavelet
                if 'cmor' in wavelet:
                    fb = float(wavelet.split('-')[1])  # bandwidth parameter
                    fc = float(wavelet.split('cmor')[1].split('-')[0])  # center frequency
                else:
                    fc = pywt.central_frequency(wavelet)
                
                # Convert frequency to scale: scale = fc * fs / frequency
                frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_scales)
                scales = fc * self.fs / frequencies
            else:
                # For other wavelets, use default scaling
                scales = np.logspace(1, 3, num_scales)
                frequencies = pywt.scale2frequency(wavelet, scales) / (1/self.fs)
        else:
            # Calculate frequencies from provided scales
            frequencies = pywt.scale2frequency(wavelet, scales) / (1/self.fs)
        
        self.scales = scales
        self.frequencies = frequencies
        
        # Compute CWT
        coefficients, _ = pywt.cwt(self.signal, scales, wavelet, sampling_period=1/self.fs)
        self.cwt_matrix = coefficients
        
        return self.cwt_matrix
    
    def compute_kurtogram(self, window_length=None, overlap=0.5):
        """Compute kurtogram from CWT coefficients"""
        if self.cwt_matrix is None:
            raise ValueError("CWT not computed. Run compute_cwt() first.")
        
        if window_length is None:
            window_length = len(self.signal) // 10
        
        step_size = int(window_length * (1 - overlap))
        num_windows = max(1, (len(self.signal) - window_length) // step_size + 1)
        
        kurtogram = np.zeros((len(self.scales), num_windows))
        time_centers = np.zeros(num_windows)
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = min(start_idx + window_length, len(self.signal))
            time_centers[i] = self.time[start_idx + (end_idx - start_idx)//2]
            
            for j in range(len(self.scales)):
                coeffs = np.abs(self.cwt_matrix[j, start_idx:end_idx])
                if len(coeffs) > 3 and np.std(coeffs) > 1e-10:
                    kurtogram[j, i] = kurtosis(coeffs, fisher=True)
                else:
                    kurtogram[j, i] = 0
        
        return kurtogram, time_centers
    
    def plot_results(self, show_kurtogram=True, figsize=(12, 10)):
        """Create comprehensive plots"""
        if show_kurtogram:
            kurtogram, time_centers = self.compute_kurtogram()
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 1, figsize=(15, 8))
            axes = [axes[0], axes[1], None, None]
        
        # Plot 1: Original Signal
        axes[0].plot(self.time, self.signal, 'b-', linewidth=1)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Original Signal')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: CWT Scalogram
        cwt_magnitude = np.abs(self.cwt_matrix)
        im1 = axes[1].imshow(cwt_magnitude, 
                           extent=[self.time[0], self.time[-1], 
                                  self.frequencies[-1], self.frequencies[0]],
                           cmap='viridis', aspect='auto', origin='upper')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_title('CWT Scalogram (Magnitude)')
        axes[1].set_yscale('log')
        plt.colorbar(im1, ax=axes[1], label='Magnitude')
        
        if show_kurtogram and len(axes) > 2:
            # Plot 3: Kurtogram
            im2 = axes[2].imshow(kurtogram,
                               extent=[time_centers[0], time_centers[-1],
                                      self.frequencies[-1], self.frequencies[0]],
                               cmap='hot', aspect='auto', origin='upper')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Frequency (Hz)')
            axes[2].set_title('Kurtogram')
            axes[2].set_yscale('log')
            plt.colorbar(im2, ax=axes[2], label='Kurtosis')
            
            # Plot 4: Maximum kurtosis vs frequency
            max_kurtosis = np.max(kurtogram, axis=1)
            axes[3].semilogx(self.frequencies, max_kurtosis, 'r-', marker='o', markersize=4)
            axes[3].set_xlabel('Frequency (Hz)')
            axes[3].set_ylabel('Maximum Kurtosis')
            axes[3].set_title('Kurtosis vs Frequency')
            axes[3].grid(True, alpha=0.3)
            
            # Find and annotate peak
            if len(max_kurtosis) > 0 and np.max(max_kurtosis) > 0:
                peak_idx = np.argmax(max_kurtosis)
                peak_freq = self.frequencies[peak_idx]
                peak_kurt = max_kurtosis[peak_idx]
                axes[3].annotate(f'Peak: {peak_freq:.1f} Hz\nKurt: {peak_kurt:.2f}',
                               xy=(peak_freq, peak_kurt), xytext=(10, 10),
                               textcoords='offset points', ha='left',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.show()
        return fig

# Example usage
def main():
    print("CWT and Kurtogram Analysis with PyWavelets")
    print("="*50)
    
    # Initialize analyzer
    analyzer = CWTKurtogramAnalyzer(fs=1000)
    
    # Test different wavelets
    wavelets = ['cmor1.5-1.0', 'morl', 'mexh']  # Complex Morlet, Morlet, Mexican Hat
    signal_types = ['chirp', 'bearing_fault', 'multi_sine']
    
    for i, (sig_type, wavelet) in enumerate(zip(signal_types, wavelets)):
        print(f"\n{i+1}. Analyzing '{sig_type}' with '{wavelet}' wavelet...")
        
        # Generate signal
        analyzer.generate_sample_signals(signal_type=sig_type, duration=2)
        
        # Compute CWT
        analyzer.compute_cwt(wavelet=wavelet, num_scales=40)
        
        # Plot results
        analyzer.plot_results(show_kurtogram=True)
        
        print(f"CWT matrix shape: {analyzer.cwt_matrix.shape}")
        print(f"Frequency range: {analyzer.frequencies[0]:.2f} - {analyzer.frequencies[-1]:.2f} Hz")

if __name__ == "__main__":
    main()