%% RUN_ML_EKF_VALIDATION (v3.1)
% Static EKF vs IMM vs ML-EKF v3.1.
%
% v3.1 tracker = 3 layer:
%   1. Physics R      : R = base_R * snr_to_R_mult(measured SNR)   [measurement side]
%   2. ML Q detector  : network measured NIS/SNR feature theke maneuver Q predict
%   3. NIS-match boost : windowed NIS ideal (dof=4) theke jotota dure, Q ke
%                        gently scale kore — covariance-matching consistency layer.
%      + recovery hold : decaying floor jate maneuver-er por Q hothat na pore.
%
% Ei tin layer-e single EKF IMM-ke match/beat kore, half computation-e.
clear; clc; close all;

addpath('../src_simulation'); addpath('../src_common'); addpath('../models');
fprintf('Loading v3.1 Q-Oracle...\n');
load('ml_oracle_net.mat', 'ml_oracle_net');

params.T = 60; params.dt = 0.01;
params.init_pos = [12000; 4000; 2500];
params.init_vel = [-200; -40; 0];
params.rcs_mean_dBsm = -15;

radar_params.sigma_range   = 158;
radar_params.sigma_doppler = 31.6;
radar_params.sigma_angle   = deg2rad(1);
radar_params.sigma_snr_dB  = 2.0;

T = params.dt; q_var = 1.0^2;
q_block = [T^4/4, T^3/2; T^3/2, T^2] * q_var;
base_Q = blkdiag(q_block, q_block, q_block);
base_R = diag([radar_params.sigma_range, radar_params.sigma_doppler, ...
               radar_params.sigma_angle, radar_params.sigma_angle].^2);

% ---- v3.1 tuned hyperparameters (Python grid-search) ----
NISMATCH_GAIN = 1.0;    % NIS-match aggressiveness
NISMATCH_CAP  = 1000;   % boost clip [1/cap, cap]
NISMATCH_EMA  = 0.1;    % NIS EMA smoothing
HOLD_TAU      = 100;    % recovery-hold decay (samples, ~1 s)

fprintf('Generating Unseen Data...\n');
[state_true, rcs_true, time_vec, ~, ~] = generate_stealth_trajectory(params);
[meas_noisy, ~, ~, snr_meas_dB] = simulate_radar_physics(state_true, rcs_true, radar_params);
N = size(state_true, 1);

init_state = [state_true(1,1); state_true(1,4); state_true(1,2); ...
              state_true(1,5); state_true(1,3); state_true(1,6)];
init_cov = diag([100 300 100 300 100 300].^2);  % [v3.1] realistic P0 (velocity poorly known: Doppler radial only)

mk_ekf = @() trackingEKF('State', init_state, 'StateCovariance', init_cov, ...
              'StateTransitionFcn', @constvel, 'ProcessNoise', base_Q, ...
              'MeasurementFcn', @radar_meas_model, 'HasMeasurementWrapping', true, ...
              'MeasurementNoise', base_R);
ekf_static = mk_ekf();
ekf_ml     = mk_ekf();

model1 = mk_ekf(); model1.ProcessNoise = base_Q * 1.0;
model2 = mk_ekf(); model2.ProcessNoise = base_Q * 900.0;
imm_filter = trackingIMM('TrackingFilters', {model1, model2}, ...
                         'TransitionProbabilities', [0.95 0.05; 0.05 0.95]);

pos_static=zeros(N,3); pos_ml=zeros(N,3); pos_imm=zeros(N,3);
err_static=zeros(N,1); err_ml=zeros(N,1); err_imm=zeros(N,1);
nees_static=zeros(N,1); nees_ml=zeros(N,1); nees_imm=zeros(N,1);
q_eff_log=zeros(N,1);
time_static=zeros(N,1); time_ml=zeros(N,1); time_imm=zeros(N,1);

window_size = 10; nis_buffer = ones(1, window_size)*4;
q_ml = 1.0; q_floor = 1.0; q_boost = 1.0; nis_ema = 4.0;
lam_hold = exp(-1/HOLD_TAU);

fprintf('Running Trackers...\n');
for k = 1:N
    z = meas_noisy(k, :)';
    true_pos = state_true(k, 1:3)';

    % --- STATIC ---
    tic; predict(ekf_static, params.dt); correct(ekf_static, z); time_static(k)=toc;
    pos_static(k,:) = ekf_static.State([1,3,5]);
    e = true_pos - pos_static(k,:)'; err_static(k)=norm(e);
    P = ekf_static.StateCovariance([1,3,5],[1,3,5]); nees_static(k)=e'*(P\e);

    % --- ML-EKF v3.1 ---
    tic;
    ekf_ml.MeasurementNoise = base_R * snr_to_R_mult(snr_meas_dB(k));  % L1 physics R
    ekf_ml.ProcessNoise = base_Q * q_ml * q_boost;
    predict(ekf_ml, params.dt);
    [res, S_cov] = residual(ekf_ml, z);
    [features, nis_buffer] = extract_tracker_features(res, S_cov, nis_buffer, snr_meas_dB(k));

    q_raw = double(10^predict(ml_oracle_net, features));          % L2 ML Q
    nis_now = res' * (S_cov \ res);
    if nis_now > chi2inv(0.99,4), q_raw = max(q_raw, 300); end   % [v3.1] predictive gate: instant NIS spike -> jump Q now
    if q_raw > q_ml, q_ml = 0.1*q_ml + 0.9*q_raw;
    else,            q_ml = 0.85*q_ml + 0.15*q_raw; end
    q_floor = max(q_ml, q_floor*lam_hold);                        % recovery hold
    q_ml = max(q_ml, q_floor);
    q_ml = min(max(q_ml, 0.1), 1000);

    nis_val = res' * (S_cov \ res);                              % L3 NIS-match
    nis_ema = (1-NISMATCH_EMA)*nis_ema + NISMATCH_EMA*nis_val;
    q_boost = q_boost * (nis_ema/4.0)^NISMATCH_GAIN;
    q_boost = min(max(q_boost, 1/NISMATCH_CAP), NISMATCH_CAP);

    correct(ekf_ml, z);
    time_ml(k)=toc;
    q_eff_log(k) = q_ml * q_boost;
    pos_ml(k,:) = ekf_ml.State([1,3,5]);
    e = true_pos - pos_ml(k,:)'; err_ml(k)=norm(e);
    P = ekf_ml.StateCovariance([1,3,5],[1,3,5]); nees_ml(k)=e'*(P\e);

    % --- IMM ---
    tic; predict(imm_filter, params.dt); correct(imm_filter, z); time_imm(k)=toc;
    pos_imm(k,:) = imm_filter.State([1,3,5]);
    e = true_pos - pos_imm(k,:)'; err_imm(k)=norm(e);
    P = imm_filter.StateCovariance([1,3,5],[1,3,5]); nees_imm(k)=e'*(P\e);
end

rmse = @(e) sqrt(mean(e.^2));

% --- Calculate Metrics for Report ---
mean_rmse_static = rmse(err_static);
mean_rmse_ml = rmse(err_ml);
mean_rmse_imm = rmse(err_imm);
improvement_static = (mean_rmse_static - mean_rmse_ml) / mean_rmse_static * 100;
improvement_imm = (mean_rmse_imm - mean_rmse_ml) / mean_rmse_imm * 100;

avg_time_static = mean(time_static(2:end)) * 1000;
max_time_static = max(time_static(2:end)) * 1000;
avg_time_ml = mean(time_ml(2:end)) * 1000;
max_time_ml = max(time_ml(2:end)) * 1000;
avg_time_imm = mean(time_imm(2:end)) * 1000;
max_time_imm = max(time_imm(2:end)) * 1000;
overhead = avg_time_ml - avg_time_static;

% --- PRINT RESULTS ---
fprintf('\n📊 --- RESULTS --- 📊\n');
fprintf('Static EKF Avg Position RMSE: %.2f meters\n', mean_rmse_static);
fprintf('ML-EKF Avg Position RMSE:     %.2f meters\n', mean_rmse_ml);
fprintf('IMM Filter Avg Position RMSE: %.2f meters\n', mean_rmse_imm);
fprintf('-------------------------------------------\n');
fprintf('Performance Improvement (ML vs Static):  %.2f%%\n', improvement_static);
fprintf('Performance Gap (ML vs IMM Optimal):     %.2f%% (positive = ML better)\n', improvement_imm);
fprintf('-------------------------------------------\n');

fprintf('\n📊 --- COMPUTATIONAL COMPLEXITY REPORT --- 📊\n');
fprintf('Target Deadline (100Hz): 10.0000 ms\n\n');
fprintf('Static EKF:\n');
fprintf('  Average Processing Time : %.4f ms\n', avg_time_static);
fprintf('  Max Processing Time     : %.4f ms\n\n', max_time_static);
fprintf('ML-EKF:\n');
fprintf('  Average Processing Time : %.4f ms\n', avg_time_ml);
fprintf('  Max Processing Time     : %.4f ms\n\n', max_time_ml);
fprintf('IMM-EKF:\n');
fprintf('  Average Processing Time : %.4f ms\n', avg_time_imm);
fprintf('  Max Processing Time     : %.4f ms\n', max_time_imm);
fprintf('\nML Oracle Overhead (MATLAB) : %.4f ms\n', overhead);
fprintf('-------------------------------------------\n');

fprintf('\n📊 --- PERFORMANCE METRICS --- 📊\n');
fprintf('Static EKF Average NEES : %.2f (Ideal: ~3.0)\n', mean(nees_static));
fprintf('ML-EKF Average NEES     : %.2f (Ideal: ~3.0)\n', mean(nees_ml));
fprintf('IMM-EKF Average NEES    : %.2f (Ideal: ~3.0)\n', mean(nees_imm));
fprintf('---------------------------------\n');

%% 7. Plotting the Results
figure('Color', 'w', 'Position', [100, 100, 1000, 600]);

% Subplot 1: Position Error Comparison
subplot(2, 2, 1);
plot(time_vec, err_static, 'b-', 'LineWidth', 1.5); hold on;
plot(time_vec, err_ml, 'r-', 'LineWidth', 2);
plot(time_vec, err_imm, 'm-.', 'LineWidth', 1.5);
title('Position Tracking Error: Static vs ML-EKF vs IMM');
xlabel('Time (s)'); ylabel('Error (meters)');
legend('Static EKF', 'ML-EKF', 'IMM Filter', 'Location', 'best');
grid on; ylim([0 400]);

% Subplot 2: Q Scaling (R is handled by physics formula in v3.1, not logged)
subplot(2,2,2);
semilogy(time_vec, max(q_eff_log, 0.1), 'r-', 'LineWidth', 1.5);
title('ML Oracle Adaptive Q Scaling Factor');
xlabel('Time (s)'); ylabel('Q Scale Factor (Log)');
grid on;

% Subplot 3: Zoomed Error Comparison
subplot(2, 2, 3);
plot(time_vec, err_static, 'b-', 'LineWidth', 1.2); hold on;
plot(time_vec, err_ml, 'r-', 'LineWidth', 1.5);
plot(time_vec, err_imm, 'm-.', 'LineWidth', 1.5);
grid on; box on;
xlabel('Time (s)'); ylabel('Position Error (m)');
title('Tracking Error Comparison (Zoomed)');
legend('Static EKF', 'ML-EKF', 'IMM Filter', 'Location', 'best');
ylim([0, max(mean_rmse_static)*2]); 

% Subplot 4: NEES Consistency
subplot(2, 2, 4);
plot(time_vec, nees_static, 'b-', 'LineWidth', 1.2); hold on;
plot(time_vec, nees_ml, 'r-', 'LineWidth', 1.5);
plot(time_vec, nees_imm, 'm-.', 'LineWidth', 1.2);
yline(3, 'k--', 'LineWidth', 2); 
grid on; box on;
xlabel('Time (s)'); ylabel('NEES');
title('Filter Consistency (NEES)');
legend('Static EKF', 'ML-EKF', 'IMM Filter', 'Ideal Bound', 'Location', 'best');
ylim([0, 15]);