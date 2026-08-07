function [meas_noisy, meas_true, snr_dB_hist, snr_meas_dB] = simulate_radar_physics(state_hist, rcs_hist, radar_params)
% SIMULATE_RADAR_PHYSICS Generates measurements with SNR-dependent noise
%
% Output:
%   meas_noisy  : [N x 4] matrix [range, doppler, azimuth, elevation]
%   meas_true   : [N x 4] matrix (noise-free ground truth)
%   snr_dB_hist : [N x 1] TRUE SNR history in dB (ground truth, plotting only!)
%   snr_meas_dB : [N x 1] MEASURED SNR in dB (noisy estimate the receiver
%                 actually observes -- eta ML feature hishebe use korbe,
%                 snr_dB_hist NA! Karon true RCS/SNR real radar jane na.)

    N = size(state_hist, 1);
    meas_noisy  = zeros(N, 4);
    meas_true   = zeros(N, 4);
    snr_dB_hist = zeros(N, 1);
    snr_meas_dB = zeros(N, 1);

    % SNR estimation error of the receiver (typical 1-3 dB)
    if isfield(radar_params, 'sigma_snr_dB')
        sigma_snr = radar_params.sigma_snr_dB;
    else
        sigma_snr = 2.0;
    end

    for k = 1:N
        % 1. True Geometry Extraction
        pos = state_hist(k, 1:3)';
        vel = state_hist(k, 4:6)';

        range_true = max(norm(pos), 1e-6);

        ux = pos / range_true;
        az_true = atan2(pos(2), pos(1));
        el_true = asin(pos(3) / range_true);
        radial_vel_true = dot(vel, ux);

        meas_true(k, :) = [range_true, radial_vel_true, az_true, el_true];

        % 2. SNR (Simplified Radar Equation): 20 dB at 10 km for 0 dBsm
        base_snr_linear = 10^(20/10);
        ref_range = 10000;

        snr_linear = base_snr_linear * rcs_hist(k) * (ref_range / range_true)^4;
        snr_linear = max(snr_linear, 1e-4);
        snr_dB = 10*log10(snr_linear);
        snr_dB_hist(k) = snr_dB;

        % 2b. MEASURED SNR (what the receiver reports, with estimation error)
        %     >>> FIX: leakage-free feature source <<<
        snr_meas_dB(k) = snr_dB + sigma_snr * randn;

        % 3. SNR-Dependent Measurement Noise (CRLB behavior: sigma ~ 1/sqrt(SNR))
        noise_multiplier = max(1.0, 1 / sqrt(snr_linear));
        noise_multiplier = min(noise_multiplier, 10.0);

        sigma_r = radar_params.sigma_range   * noise_multiplier;
        sigma_d = radar_params.sigma_doppler * noise_multiplier;
        sigma_a = radar_params.sigma_angle   * noise_multiplier;

        % 4. Add Gaussian Noise (azimuth wrapped to [-pi, pi])
        meas_noisy(k, 1) = range_true      + sigma_r * randn;
        meas_noisy(k, 2) = radial_vel_true + sigma_d * randn;
        meas_noisy(k, 3) = wrapToPi(az_true + sigma_a * randn);
        meas_noisy(k, 4) = el_true         + sigma_a * randn;
    end
end
