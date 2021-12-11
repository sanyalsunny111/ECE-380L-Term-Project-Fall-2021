import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats

import complex_pca


def plot_pca_variance_curve(x: np.ndarray, title: str = 'PCA -- Variance Explained Curve') -> None:
    pca = complex_pca.ComplexPCA(n_components=x.shape[1])
    pca.fit(x)

    plt.figure()
    plt.plot(range(1, x.shape[1] + 1), np.cumsum(pca.explained_variance_ratio_) / np.sum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Proportion of Variance Captured')
    plt.title(title)
    plt.grid(True)


# noinspection DuplicatedCode
def main() -> None:
    # data_path = r'D:\EE 364D\dataset\synthetic_data\channel_specific\train_indoor\subsampled\10_percent\train_indoor_channel_e_flat_3.h5'
    data_path = r'D:\EE 364D\dataset\synthetic_data\channel_specific\test_indoor_20dB\test_indoor_20dB_channel_e_flat.h5'
    constant_features_path = '../data_preprocessing/constant_features.mat'

    data = h5py.File(data_path, 'r')
    constant_features = scipy.io.loadmat(constant_features_path, squeeze_me=True)
    constant_features = constant_features['constant']

    # Number of data points to use.
    n = 1

    # Data and pilot extraction.
    data_indices = constant_features['iMDataTone_HePpdu'][()].astype(np.int32) - 1
    pilot_indices = constant_features['iMPilotTone_HePpdu'][()].astype(np.int32) - 1
    data_size = 256

    rx_pilot = np.array(data['rx_pilot'][0:n, :])
    tx_pilot = np.array(data['tx_pilot'][0:n, :])
    pilot_gain = rx_pilot / tx_pilot

    rx_data = np.array(data['rx_data'][0:n, :])
    tx_data = np.array(data['tx_data'][0:n, :])
    data_gain = rx_data / tx_data

    # L-LTF extraction.
    l_ltf_size = 64

    rx_l_ltf_1 = np.array(data['rx_l_ltf_1'][0:n, :])
    rx_l_ltf_2 = np.array(data['rx_l_ltf_2'][0:n, :])

    tx_l_ltf = constant_features['txLltfFftOut'][()]

    rx_l_ltf_1_trimmed = rx_l_ltf_1[:, tx_l_ltf != 0]
    rx_l_ltf_2_trimmed = rx_l_ltf_2[:, tx_l_ltf != 0]
    tx_l_ltf_trimmed = tx_l_ltf[tx_l_ltf != 0]

    l_ltf_1_trimmed_gain = rx_l_ltf_1_trimmed / tx_l_ltf_trimmed
    l_ltf_2_trimmed_gain = rx_l_ltf_2_trimmed / tx_l_ltf_trimmed

    # HE-LTF extraction.
    he_ltf_data_indices = constant_features['iMDataTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_pilot_indices = constant_features['iMPilotTone_Heltf'][()].astype(np.int32) - 1
    he_ltf_size = 256

    rx_he_ltf_data = np.array(data['rx_he_ltf_data'][0:n, :])
    rx_he_ltf_pilot = np.array(data['rx_he_ltf_pilot'][0:n, :])
    rx_he_ltf = np.zeros((rx_he_ltf_data.shape[0], he_ltf_size), dtype=complex)
    rx_he_ltf[:, he_ltf_data_indices] = rx_he_ltf_data
    rx_he_ltf[:, he_ltf_pilot_indices] = rx_he_ltf_pilot

    tx_he_ltf = constant_features['txHeltfFftOut'][()]

    rx_he_ltf_trimmed = rx_he_ltf[:, tx_he_ltf != 0]
    tx_he_ltf_trimmed = tx_he_ltf[tx_he_ltf != 0]

    he_ltf_trimmed_gain = rx_he_ltf_trimmed / tx_he_ltf_trimmed

    # Frequency domain.
    f = np.linspace(0, 1, data_size)
    f_data = f[data_indices]
    f_pilot = f[pilot_indices]

    f_rx_he_ltf = np.linspace(0, 1, he_ltf_size)
    f_rx_he_ltf_trimmed = f_rx_he_ltf[tx_he_ltf != 0]

    f_l_ltf = np.linspace(0, 1, l_ltf_size)
    f_l_ltf_trimmed = f_l_ltf[tx_l_ltf != 0]

    # Channel instance to use.
    i = 0

    # Make plots.
    plot_constellation = False
    plot_magnitude = True
    plot_phase = True
    plot_pca = False
    plot_mean_magnitude = False
    plot_correction_phase = False

    if plot_constellation:
        plt.figure()
        plt.scatter(np.real(tx_he_ltf_trimmed), np.imag(tx_he_ltf_trimmed))
        plt.scatter(np.real(tx_l_ltf_trimmed), np.imag(tx_l_ltf_trimmed))
        plt.scatter(np.real(tx_pilot[i, :]), np.imag(tx_pilot[i, :]))
        plt.xlabel('In-phase Component')
        plt.ylabel('Quadrature Component')
        plt.title('Transmitted Symbol Constellation')
        plt.legend(['HE-LTF', 'L-LTF-1', 'L-LTF-2', 'Pilot'])
        plt.grid()

        plt.figure()
        plt.scatter(np.real(he_ltf_trimmed_gain[i, :]), np.imag(he_ltf_trimmed_gain[i, :]))
        plt.scatter(np.real(l_ltf_1_trimmed_gain[i, :]), np.imag(l_ltf_1_trimmed_gain[i, :]))
        plt.scatter(np.real(l_ltf_2_trimmed_gain[i, :]), np.imag(l_ltf_2_trimmed_gain[i, :]))
        plt.scatter(np.real(pilot_gain[i, :]), np.imag(pilot_gain[i, :]))
        plt.xlabel('In-phase Component')
        plt.ylabel('Quadrature Component')
        plt.title('Channel Gain Estimate Constellation')
        plt.legend(['HE-LTF', 'L-LTF-1', 'L-LTF-2', 'Pilot'])
        plt.grid()

    if plot_magnitude:
        plt.figure()
        plt.scatter(f_rx_he_ltf_trimmed, 20 * np.log10(np.abs(he_ltf_trimmed_gain[i, :])))
        plt.scatter(f_l_ltf_trimmed, 20 * np.log10(np.abs(l_ltf_1_trimmed_gain[i, :])))
        plt.scatter(f_l_ltf_trimmed, 20 * np.log10(np.abs(l_ltf_2_trimmed_gain[i, :])))
        plt.scatter(f_pilot, 20 * np.log10(np.abs(pilot_gain[i, :])))
        plt.scatter(f_data, 20 * np.log10(np.abs(data_gain[i, :])), marker='x')
        plt.xlabel(r'$f$ (normalized)')
        plt.ylabel(r'$|H|^2$ (dB)')
        plt.title('Channel Gain Estimate')
        plt.legend(['HE-LTF', 'L-LTF-1', 'L-LTF-2', 'Pilot', 'Data'])
        plt.grid()

    if plot_phase:
        plt.figure()
        unwrap = False
        if unwrap:
            plt.scatter(f_rx_he_ltf_trimmed, np.unwrap(np.angle(he_ltf_trimmed_gain[i, :])) / np.pi)
            plt.scatter(f_l_ltf_trimmed, np.unwrap(np.angle(l_ltf_1_trimmed_gain[i, :])) / np.pi)
            plt.scatter(f_l_ltf_trimmed, np.unwrap(np.angle(l_ltf_2_trimmed_gain[i, :])) / np.pi)
            plt.scatter(f_pilot, np.unwrap(np.angle(pilot_gain[i, :])) / np.pi)
            plt.scatter(f_data, np.unwrap(np.angle(data_gain[i, :])) / np.pi, marker='x')
        else:
            plt.scatter(f_rx_he_ltf_trimmed, np.angle(he_ltf_trimmed_gain[i, :]) / np.pi)
            plt.scatter(f_l_ltf_trimmed, np.angle(l_ltf_1_trimmed_gain[i, :]) / np.pi)
            plt.scatter(f_l_ltf_trimmed, np.angle(l_ltf_2_trimmed_gain[i, :]) / np.pi)
            plt.scatter(f_pilot, np.angle(pilot_gain[i, :]) / np.pi)
            plt.scatter(f_data, np.angle(data_gain[i, :]) / np.pi, marker='x')
        plt.xlabel(r'$f$ (normalized)')
        plt.ylabel(r'$\angle H$ ($\times \pi^{-1}$)')
        plt.title('Channel Phase')
        plt.legend(['HE-LTF', 'L-LTF-1', 'L-LTF-2', 'Pilot', 'Data'])
        plt.grid()

    if plot_pca:
        plot_pca_variance_curve(he_ltf_trimmed_gain, 'HE-LTF Trimmed Gain')
        plot_pca_variance_curve(rx_he_ltf, 'HE-LTF Raw')
        plot_pca_variance_curve(l_ltf_1_trimmed_gain, 'L-LTF-1 Trimmed Gain')
        plot_pca_variance_curve(rx_l_ltf_1, 'L-LTF-1 Raw')
        plot_pca_variance_curve(l_ltf_2_trimmed_gain, 'L-LTF-2 Trimmed Gain')
        plot_pca_variance_curve(rx_l_ltf_2, 'L-LTF-2 Raw')
        plot_pca_variance_curve(rx_pilot, 'Pilot Raw')
        plot_pca_variance_curve(pilot_gain, 'Pilot Gain')
        plot_pca_variance_curve(np.hstack([
            he_ltf_trimmed_gain,
            l_ltf_1_trimmed_gain,
            l_ltf_2_trimmed_gain,
            pilot_gain
        ]), 'HE-LTF, L-LTF-1, L-LTF-2, and Pilot Trimmed Gain')

    if plot_mean_magnitude:
        plt.figure()
        x = f_rx_he_ltf_trimmed
        y = np.mean(np.abs(he_ltf_trimmed_gain), axis=0)
        s = np.std(np.abs(he_ltf_trimmed_gain), axis=0)
        plt.plot(x, 20 * np.log10(y))
        plt.fill_between(x, 20 * np.log10(y - s), 20 * np.log10(y + s), alpha=0.5)
        plt.xlabel(r'$f$ (normalized)')
        plt.ylabel(r'$|H|^2$ (dB)')
        plt.title('Mean Channel Gain')
        plt.legend([r'$\mu$', r'$\pm\sigma$'])
        plt.grid()

    if plot_correction_phase:
        index = np.arange(0, he_ltf_size)[tx_he_ltf != 0]
        phase = np.angle(he_ltf_trimmed_gain[0, :])
        consecutive_phase = np.split(phase, np.where(np.diff(index) != 1)[0] + 1)
        consecutive_index = np.split(index, np.where(np.diff(index) != 1)[0] + 1)
        consecutive_phase = [np.unwrap(x) for x in consecutive_phase]
        consecutive_fits = [scipy.stats.linregress(x, y) for x, y in zip(consecutive_index, consecutive_phase)]

        combined_phase = []
        for x, y in zip(consecutive_index, consecutive_phase):
            y_hat = x * consecutive_fits[0].slope + consecutive_fits[0].intercept
            # We can add this offset WLoG because phase is 2π periodic.
            offset = 2 * np.pi * np.round((y_hat - y) / (2 * np.pi))
            combined_phase.append(y + offset)

        combined_phase = np.hstack(combined_phase)

        plt.figure()
        for x, y in zip(consecutive_index, consecutive_phase):
            plt.scatter(x, y / np.pi)

        for fit in consecutive_fits:
            x = np.linspace(0, he_ltf_size, 1000)
            y = fit.slope * x + fit.intercept
            plt.plot(x, y / np.pi)

        plt.xlabel('Subcarrier Index')
        plt.ylabel(r'$\angle H$ ($\times \pi^{-1}$)')
        plt.title('HE-LTF Channel Phase Estimates')
        plt.legend([f'Interval {i + 1}' for i in range(len(consecutive_index))])
        plt.grid()

        plt.figure()
        plt.scatter(index, combined_phase / np.pi)
        plt.xlabel('Subcarrier Index')
        plt.ylabel(r'$\angle H$ ($\times \pi^{-1}$)')
        plt.title('HE-LTF Channel Phase Combined Estimate')
        plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
