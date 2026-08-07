%% GENERATE_OPTIMIZED_DATASET (v3.0)
% v3.0 architecture change:
%  - R ekhon PHYSICS diye set hoy (snr_to_R_mult), ML diye na.
%  - ML ekhon SHUDHU Q predict kore (single output) — maneuver detection.
%  - Label = ramp-smoothed maneuver flag * log10(Q_MANEUVER). Recovery hold
%    ekhon DEPLOYMENT-side (NIS-match layer) handle kore.
%
% Python experiment-e proven (20-run Monte Carlo):
%   Static  RMSE 291 / ANEES 4530
%   v2.3    RMSE  98 / ANEES   77
%   v3.0    RMSE  28 / ANEES  4.6  (median NEES 2.6 — ideal!)
clear; clc;
addpath('../src_simulation');
addpath('../src_common');

N_MC = 30;
params.T = 60; params.dt = 0.01;
params.init_pos = [15000; 5000; 3000];
params.init_vel = [-250; -50; 0];
params.rcs_mean_dBsm = -15;

radar_params.sigma_range   = 50;
radar_params.sigma_doppler = 10;
radar_params.sigma_angle   = deg2rad(1);
radar_params.sigma_snr_dB  = 2.0;

T = params.dt; q_var = 1.0^2;
q_block = [T^4/4, T^3/2; T^3/2, T^2] * q_var;
base_Q = blkdiag(q_block, q_block, q_block);
base_R = diag([radar_params.sigma_range, radar_params.sigma_doppler, ...
               radar_params.sigma_angle, radar_params.sigma_angle].^2);

Q_MANEUVER = 900.0; Q_CRUISE = 1.0; RAMP_STEPS = 30;
NUM_FEATURES = 4; window_size = 10;

total_samples = N_MC * (floor(params.T/params.dt) + 1);
X_train = zeros(total_samples, NUM_FEATURES);
Y_train = zeros(total_samples, 1);    % single output: log10(Q_scale)
run_id  = zeros(total_samples, 1);
sample_idx = 1;

fprintf('v3.0 Dataset Generation (physics-R, Q-only oracle)...\n');
for mc = 1:N_MC
    fprintf('   Run %d/%d...\n', mc, N_MC);
    [state_true, rcs_true, ~, maneuver_flag] = generate_stealth_trajectory(params);
    [meas_noisy, ~, ~, snr_meas_dB] = simulate_radar_physics(state_true, rcs_true, radar_params);
    N = size(state_true, 1);
    soft_flag = movmean(double(maneuver_flag), RAMP_STEPS);
    init_state = [state_true(1,1); state_true(1,4); state_true(1,2); ...
                  state_true(1,5); state_true(1,3); state_true(1,6)];
    ekf_obj = trackingEKF('State', init_state, 'StateCovariance', eye(6)*500^2, ...
                          'StateTransitionFcn', @constvel, 'ProcessNoise', base_Q, ...
                          'MeasurementFcn', @radar_meas_model, ...
                          'HasMeasurementWrapping', true, 'MeasurementNoise', base_R);
    nis_buffer = ones(1, window_size) * 4;
    for k = 2:N
        z = meas_noisy(k, :)';
        ekf_obj.MeasurementNoise = base_R * snr_to_R_mult(snr_meas_dB(k)); % physics R
        predict(ekf_obj, params.dt);
        [res, S_cov] = residual(ekf_obj, z);
        [features, nis_buffer] = extract_tracker_features(res, S_cov, nis_buffer, snr_meas_dB(k));
        X_train(sample_idx, :) = features;
        alpha = soft_flag(k);
        Y_train(sample_idx) = (1-alpha)*log10(Q_CRUISE) + alpha*log10(Q_MANEUVER);
        run_id(sample_idx) = mc;
        ekf_obj.ProcessNoise = base_Q * 10^Y_train(sample_idx);
        correct(ekf_obj, z);
        sample_idx = sample_idx + 1;
    end
end
X_train = X_train(1:sample_idx-1, :);
Y_train = Y_train(1:sample_idx-1, :);
run_id  = run_id(1:sample_idx-1);
save('../data/ml_training_set.mat', 'X_train', 'Y_train', 'run_id');
fprintf('v3.0 Dataset Saved (%d samples)\n', size(X_train,1));
