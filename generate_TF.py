import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sgn
import time

ANALOG_ATTN = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9]  # in dB
DIGITAL_ATTN = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9]  # in dB
WRF = 2518300671.117578  #: RF frequency (angular)
NP = 662
FMAX = 1.619443e6

# The parameters are:
# phase: phi (-20, 20), with step 1
# gain: dgoo: (1e-4, 1e-2), 20 steps in total

def low_pass_filter(signal, cutoff_frequency=0.5):
    """Low-pass filter based on Butterworth 5th order digital filter from
    scipy,
    http://docs.scipy.org

    Parameters
    ----------
    signal : float array
        Signal to be filtered
    cutoff_frequency : float
        Cutoff frequency [1] corresponding to a 3 dB gain drop, relative to the
        Nyquist frequency of 1; default is 0.5

    Returns
    -------
    float array
        Low-pass filtered signal

    """

    b, a = sgn.butter(5, cutoff_frequency, 'low', analog=False)

    return sgn.filtfilt(b, a, signal)


class GenerateTF(object):

    def __init__(self, fb_attn_index=3, with_noise=False):
        self.fb_attn_index = fb_attn_index  # depends on cavity
        self.analog_attn = ANALOG_ATTN[self.fb_attn_index]
        self.digital_attn = DIGITAL_ATTN[self.fb_attn_index]
        self.frequency = np.linspace(-FMAX, FMAX, NP)
        self.with_noise = with_noise
        # self.closed_loop_response(self.frequency)
        # self.noise = self.add_noise(relative_amplitude=0.2)
        # self.cl_response += self.noise
        # if plot:
        #     self.bode_plot()

    def __call__(self, frequency, phi=0, g_oo=2e-3):
        self.closed_loop_response(frequency, phi, g_oo)
        if self.with_noise:
            self.noise = self.add_noise(relative_amplitude=0.2)
            self.cl_response += self.noise
        amplitude_linear = np.absolute(self.cl_response)
        amplitude_dB = 20 * np.log10(amplitude_linear)
        return amplitude_dB

    def closed_loop_response(self, frequency, phi, g_oo):
        # Use measured value
        #        g_a = prm.analog.gain*analog_attn
        #        g_d = (prm.digital.gain/g_a)*digital_gain*digital_attn

        feedback_model = self.feedback_TF(frequency,
                                          tau_a=170e-6,  # prm.analog.tau,
                                          tau_d=400e-6,  # prm.digital.tau,
                                          g_a=2,  # 6.8e-6, #g_a,
                                          g_d=10,  # g_d
                                          #            delta_phi=prm.digital.phase,
                                          )

        cavity_model = self.cavity_TF(frequency,
                                      delta_w=6e3,  # prm.cavity.wr - self.WRF,
                                      g_oo=g_oo,  # init guess, was 1e-2 # range 2e-2 to 5e-4
                                      q=60000,  # q_cl,
                                      w_rf=WRF
                                      )

        phase_model = self.phase_TF(frequency,
                                    phi=phi,  # init guess, was 50
                                    tau=650e-9  # prm.delay.loop,
                                    )

        tf_model = feedback_model * cavity_model * phase_model
        self.cl_response = tf_model / (1 - tf_model)

    def add_noise(self, relative_amplitude, filter=True):
        maximum = np.max(np.absolute(self.cl_response))

        np.random.seed(np.uint32(time.time_ns()))
        noise_real = np.random.rand(NP) - 0.5
        noise_imag = np.random.rand(NP) - 0.5

        if filter:
            cutoff = 0.5  # 1e4*2.*2e-4
            noise_real = low_pass_filter(noise_real, cutoff_frequency=cutoff)
            noise_imag = low_pass_filter(noise_imag, cutoff_frequency=cutoff)

        #        self.cl_response += (noise_real + 1j*noise_imag)*maximum*relative_amplitude
        return (noise_real + 1j * noise_imag) * maximum * relative_amplitude

    def feedback_TF(self, frequency, tau_a, tau_d, g_a, g_d, delta_phi=0):
        w = 2 * np.pi * frequency  # omega

        # fraction 1: \\frac{G_{D} e^{\\Delta\\varphi \\frac{\\pi}{180} i}}{
        # 1+\\omega \\tau_{D} i}
        # Digital FB
        num1 = g_d * np.exp(np.radians(delta_phi) * 1j)
        den1 = 1 + w * tau_d * 1j

        # fraction 2: \\frac{\\tau_{A} \\omega}{\\tau_{A} \\omega-i}
        # Analog FB
        num2 = tau_a * w
        den2 = tau_a * w - 1j

        resp = -g_a * (num1 / den1 + num2 / den2)

        return resp

    def cavity_TF(self, frequency, delta_w, g_oo, q, w_rf, r_over_q=45):
        w = 2 * np.pi * frequency  # omega
        s = 1j * (w + w_rf)

        w_r = w_rf + delta_w  # resonant frequency

        resp = g_oo * r_over_q * w_r * s / np.sqrt(q) / (
                s ** 2 + (w_r / q) * s + w_r ** 2)

        return resp

    def phase_TF(self, frequency, phi, tau):
        w = 2 * np.pi * frequency  # omega

        resp = np.exp(phi * np.pi / 180 * 1j) * np.exp((-1) * w * tau * 1j)

        return resp

    def bode_plot(self):
        amplitude_linear = np.absolute(self.cl_response)
        phase = np.angle(self.cl_response)

        amplitude_dB = 20 * np.log10(amplitude_linear)

        plt.figure("Amplitude")
        plt.plot(self.frequency, amplitude_dB)

        phase = np.unwrap(phase) / np.pi * 180
        phase -= np.mean(phase)
        plt.figure("Phase")
        plt.plot(self.frequency, phase)
        plt.show()


if __name__ == "__main__":
    gen_tf = GenerateTF(fb_attn_index=3)
    gen_tf(gen_tf.frequency)
    gen_tf.bode_plot()
