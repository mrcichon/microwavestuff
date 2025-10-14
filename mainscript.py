import skrf as rf
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d

# Apply scikit-rf > 0.17.0 compatibility patch for time_gate functionality.
# This must run before the App is instantiated.
if rf.__version__ > "0.17.0":
    def find_nearest_index(array, value):
        return (np.abs(array-value)).argmin()

    def delay_v017(self, d, unit='ns', port=0, media=None, **kw):
        if d == 0: return self
        d = d/2.
        if self.nports > 2:
            raise NotImplementedError('only implemented for 1 and 2 ports')
        if media is None:
            from skrf.media import Freespace
            media = Freespace(frequency=self.frequency, z0=self.z0[:,port])
        l = media.line(d=d, unit=unit, **kw)
        return l**self

    def time_gate_v017(self, center=None, span=None, **kwargs):
        if self.nports > 1:
            raise ValueError('Time-gating only works on one-ports')

        window = kwargs.get('window', ('kaiser', 6))
        mode = kwargs.get('mode', 'bandpass')
        boundary = kwargs.get('boundary', 'reflect')
        
        center_s = center * 1e-9
        span_s = span * 1e-9
        start, stop = center_s - span_s/2., center_s + span_s/2.

        t = np.linspace(-0.5/self.frequency.step, 0.5/self.frequency.step, self.frequency.npoints)
        start_idx, stop_idx = find_nearest_index(t, start), find_nearest_index(t, stop)
        window_width = abs(stop_idx - start_idx)
        window_array = signal.get_window(window, window_width) if window_width > 0 else np.array([])
        
        gate = np.r_[np.zeros(start_idx), window_array, np.zeros(len(t) - stop_idx)]
        kernel = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(gate, axes=0), axis=0))
        kernel = abs(kernel).flatten()
        kernel /= sum(kernel)

        out = self.copy()
        if center != 0: out = delay_v017(out, -center, 'ns', port=0, media=kwargs.get('media'))

        re = np.real(out.s[:,0,0]); im = np.imag(out.s[:,0,0])
        s = convolve1d(re, kernel, mode=boundary) + 1j*convolve1d(im, kernel, mode=boundary)
        out.s[:,0,0] = s

        if center != 0: out = delay_v017(out, center, 'ns', port=0, media=kwargs.get('media'))
        if mode == 'bandstop': out = self - out
        return out

    rf.Network.delay = delay_v017
    rf.Network.time_gate = time_gate_v017

# --- Application Entry Point ---
if __name__ == "__main__":
    # The App class is assumed to be in the implemented ui_main.py
    from ui_main import App
    
    app = App()
    app.mainloop()
