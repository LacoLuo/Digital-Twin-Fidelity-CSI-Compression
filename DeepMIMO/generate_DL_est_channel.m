clearvars
clc
rng("default")

%% Setup system parameters
load("./DeepMIMO_datasets/Boston5G_3p5_real/channel.mat")

num_tx = 32;
total_user = size(all_channel, 1);

bandwidth = 30e3; % 5G NR sub-6 subcarrier spacing
noise_level_subcarrier = physconst('Boltzmann') * (25 + 273.15) * bandwidth; % power in watt

bs_eirp = 43; % dbm
ue_noise_figure = 7; % dB noise figure

bs_power_dbm = bs_eirp - 10*log10(num_tx); % Transmit power + antenna gain
noise_level_subcarrier_downlink = 10^((10*log10(noise_level_subcarrier) + ue_noise_figure)/10);
deepmimo_tx_dbm = 30;

downlink_channel_scaling = 10^((bs_power_dbm - deepmimo_tx_dbm)/10);

%% Channel estimation
channel = double(squeeze(all_channel(:, :, 1:num_tx, :))); 

num_sample = size(channel, 1);
num_subcarrier = size(channel, 3);

channel_est = zeros(size(all_channel));
H_est_nmse_all = zeros(num_sample, num_subcarrier);
for i = 1:num_sample
    disp(i)
    for j = 1:num_subcarrier
        sampled_channel = channel(i, :, j);
        H = sampled_channel * sqrt(downlink_channel_scaling);

        % Downlink channel estimation 
        P = eye(num_tx); % Independent pilot

        pilot_noise_shape = size(H * P);
        pilot_noise = sqrt(noise_level_subcarrier_downlink/2) * (randn(pilot_noise_shape) + 1j * randn(pilot_noise_shape)); % Total transmit power / total noise power = SNR

        y_p = H * P + pilot_noise;
        H_downlink_est = y_p * pinv(P);
        
        H_est = H_downlink_est;
        H_est_nmse = sum(abs(H_est - H).^2, 2) ./ sum(abs(H_est).^2, 2);

        channel_est(i, :, :, j) = H_est;
        H_est_nmse_all(i, j) = H_est_nmse;
    end
end
H_est_nmse_all_ = mean(H_est_nmse_all, 2);

figure(1);
tmp = H_est_nmse_all_;
ecdf(10*log10(tmp(:)));
grid on;
box on;
ylabel('CDF');
xlabel('Channel Estimation NMSE (dB)');

%% Generate angle-delay channels
channel_est = single(channel_est);
all_channel_d = ifft(channel_est, size(channel_est, 4), 4);
all_channel_d_clip = all_channel_d(:, :, :, 1:32);
all_channel_ad_clip = fftshift(fft(all_channel_d_clip, size(channel_est, 3), 3), 3);

all_channel_d_clip_ = ifft(ifftshift(all_channel_ad_clip,3), size(all_channel,3), 3);
all_channel_recover = fft(all_channel_d_clip_, size(all_channel_d, 4), 4);

