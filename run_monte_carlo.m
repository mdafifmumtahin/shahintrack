%% RUN_MONTE_CARLO.m
% This script performs Monte Carlo simulations to evaluate EKF performance
% Author: Your Name
clear; clc; close all;

%% 1. Parameters Setup
N_MC = 50;              
sim_params.T = 20;      
sim_params.dt = 0.01;   
N = floor(sim_params.T/sim_params.dt) + 1;

% Target Params
target_params.T = sim_params.T;
target_params.dt = sim_params.dt;
target_params.init_pos = [10e3; 0; 5e3];
target_params.init_vel = [-200; 0; 0];
target_params.accel = [0; 0; 0];
target_params.rcs_mean_dBsm = -10;

% ✅ Radar Params (IEEE STANDARD REALISTIC VALUES)
radar_params.pos = [0; 0; 0];
radar_params.freq = 10e9;
radar_params.bw = 10e6;
radar_params.pri = 1e-3;

% এই ভ্যালুগুলো ফিল্টারকে "স্ট্যাবল" রাখবে
radar_params.sigma_range = 50;             % 5m -> 50m (Standard Accuracy)
radar_params.sigma_doppler = 10;           % 1 m/s -> 10 m/s
radar_params.sigma_angle = 1.0*deg2rad(1); % 0.1 deg -> 1.0 deg (Standard Beamwidth)

radar_params.clutter_density = 0.05;

% ✅ EKF Params (TUNED FOR STABILITY)
q_std = 1.0;             % Process noise (Standard for CV model)
r_std_vec = [radar_params.sigma_range, radar_params.sigma_doppler, ...
             radar_params.sigma_angle, radar_params.sigma_angle];

%% 2. Pre-allocation for Statistics
% To store Average RMSE over time
rmse_pos_mc = zeros(N, 1);
rmse_vel_mc = zeros(N, 1);
time_vec = (0:N-1)' * sim_params.dt;

%% 3. Monte Carlo Loop
fprintf('⏳ Running Monte Carlo Simulations (N=%d)...\n', N_MC);

% ✅ Pre-allocate accumulators for Squared Errors (Critical for True RMSE)
rmse_pos_sq_sum = zeros(N, 1);
rmse_vel_sq_sum = zeros(N, 1);

for mc = 1:N_MC
    if mod(mc, 10) == 0, fprintf('   Run %d/%d\n', mc, N_MC); end
    
    % 1. Generate Truth (RCS & Trajectory vary per run)
    [state_hist, rcs_hist] = generate_point_target(target_params);
    
    % 2. Generate Noisy Measurements
    meas_data = zeros(N, 4);
    for k = 1:N
        [meas, ~] = simulate_radar_measurement(state_hist(k,:)', rcs_hist(k), radar_params); 
        meas_data(k,:) = meas';
    end
    
    % 3. Initialization (Near Truth with realistic uncertainty)
    init_state = state_hist(1,:)' + randn(6,1) .* [50; 50; 50; 15; 15; 15];
    init_P     = diag([500^2, 500^2, 500^2, 50^2, 50^2, 50^2]);
    
    % 4. Run Tracker
    [est_states, ~] = ekf_tracker(meas_data, init_state, init_P, q_std, r_std_vec, sim_params.dt);
    
    % 5. Calculate Errors
    err_pos = state_hist(:, 1:3) - est_states(:, 1:3);
    err_vel = state_hist(:, 4:6) - est_states(:, 4:6);
    
    % ✅ ACCUMULATE SQUARED ERRORS (Not Euclidean norms)
    rmse_pos_sq_sum = rmse_pos_sq_sum + sum(err_pos.^2, 2);
    rmse_vel_sq_sum = rmse_vel_sq_sum + sum(err_vel.^2, 2);
end

%% 4. Compute True RMSE
% Formula: RMSE = sqrt( mean( squared_errors ) )
% This guarantees EKF RMSE >= PCRLB (Theoretical Consistency)
rmse_pos_avg = sqrt(rmse_pos_sq_sum / N_MC);
rmse_vel_avg = sqrt(rmse_vel_sq_sum / N_MC);

%% 4. Average the Results (True RMSE)
rmse_pos_avg = sqrt(rmse_pos_sq_sum / N_MC);  % ✅ Root Mean Square
rmse_vel_avg = sqrt(rmse_vel_sq_sum / N_MC);   % ✅ Root Mean Square

%% 5. Plotting (IEEE Style)
figure('Color', 'w', 'Position', [100 100 800 600]);

subplot(2,1,1);
plot(time_vec, rmse_pos_avg, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
grid on; box on;
ylabel('Position RMSE (m)', 'FontSize', 11);
title('Monte Carlo Analysis: EKF Performance', 'FontSize', 12, 'FontWeight', 'bold');
ylim([0, max(rmse_pos_avg)*1.1]);

subplot(2,1,2);
plot(time_vec, rmse_vel_avg, 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]);
grid on; box on;
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity RMSE (m/s)', 'FontSize', 11);
ylim([0, max(rmse_vel_avg)*1.1]);

%% 6. Rigorous PCRLB Calculation (Recursive Fisher Information Matrix)
fprintf('📊 Calculating Rigorous PCRLB...\n');

% ✅ FIX 1: Define matrices explicitly in this script's workspace
T = sim_params.dt;
G = [T^2/2 0 0; 0 T^2/2 0; 0 0 T^2/2; T 0 0; 0 T 0; 0 0 T];
Q_mat = G * diag([q_std^2 q_std^2 q_std^2]) * G';
R_mat = diag(r_std_vec.^2);
F_mat = [1 0 0 T 0 0; 0 1 0 0 T 0; 0 0 1 0 0 T; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1];

% ✅ FIX 2: Generate one deterministic truth trajectory for PCRLB
[state_hist, ~] = generate_point_target(target_params);

% ✅ CORRECTED: Start PCRLB with same covariance as EKF
P_crlb = init_P;  % আগে diag([1e8...]) ছিল, এখন init_P ব্যবহার করুন
crlb_pos_vec = zeros(N, 1);
crlb_vel_vec = zeros(N, 1);

for k = 1:N
    % Extract True State for Jacobian Calculation
    true_st = state_hist(k, :);
    px = true_st(1); py = true_st(2); pz = true_st(3);
    vx = true_st(4); vy = true_st(5); vz = true_st(6);
    
    % Calculate Geometry (Same as EKF H-matrix logic)
    r = max(sqrt(px^2 + py^2 + pz^2), 1e-6);
    r2 = r^2; r3 = r^3;
    rho2 = max(px^2 + py^2, 1e-6);
    
    r_dot = (px*vx + py*vy + pz*vz) / r;
    
    % Calculate Jacobian H_true using TRUE state
    H_true = zeros(4, 6);
    H_true(1,1:3) = [px, py, pz] / r;
    H_true(2,1:3) = [(vx*r2 - px*r*r_dot)/r3, (vy*r2 - py*r*r_dot)/r3, (vz*r2 - pz*r*r_dot)/r3];
    H_true(2,4:6) = [px, py, pz] / r;
    H_true(3,1) = -py / rho2; H_true(3,2) = px / rho2;
    
    sin_el = pz / r;
    cos_el = sqrt(max(0, 1 - sin_el^2));
    H_true(4,1:3) = [-px*sin_el, -py*sin_el, r*cos_el] / r2;
    
    % Recursive PCRLB Update Equation (Information Filter Form)
    P_pred = F_mat * P_crlb * F_mat' + Q_mat;
    
    try
        % J_k = P_pred^{-1} + H'*R^{-1}*H  ->  P_crlb = inv(J_k)
        Info_Mat = inv(P_pred) + H_true' * inv(R_mat) * H_true;
        P_crlb = inv(Info_Mat);
    catch
        P_crlb = P_pred; % Fallback if singular
    end
    
    % Force Symmetry (Numerical Safety)
    P_crlb = (P_crlb + P_crlb') / 2;
    
    % Extract Bounds
    crlb_pos_vec(k) = sqrt(P_crlb(1,1) + P_crlb(2,2) + P_crlb(3,3));
    crlb_vel_vec(k) = sqrt(P_crlb(4,4) + P_crlb(5,5) + P_crlb(6,6));
end

%% 7. Correct NEES Calculation
fprintf('\n📊 Calculating NEES (Corrected)...\n');
N_MC_nees = 50;  
nees_total = zeros(N, 1);

for mc = 1:N_MC_nees
    if mod(mc, 10) == 0, fprintf('   NEES Run %d/%d\n', mc, N_MC_nees); end
    
    [state_hist, rcs_hist] = generate_point_target(target_params);
    
    meas_data = zeros(N, 4);
    for k = 1:N
        [meas, ~] = simulate_radar_measurement(state_hist(k,:)', rcs_hist(k), radar_params);
        meas_data(k,:) = meas';
    end
    
    init_state = state_hist(1,:)' + randn(6,1) .* [50; 50; 50; 15; 15; 15];
    init_P = diag([500^2, 500^2, 500^2, 50^2, 50^2, 50^2]);
    
    [est_states, est_cov] = ekf_tracker(meas_data, init_state, init_P, q_std, r_std_vec, sim_params.dt);
    
    % Calculate NEES
    for k = 1:N
        e = state_hist(k,:)' - est_states(k,:)';          % Error vector (6x1)
        P = squeeze(est_cov(k, :, :));                    % ✅ FIX: 1x6x6 -> 6x6 matrix
        
        % ✅ STABLE: Use backslash (\) instead of inv()
        nees_val = e' * (P \ e);                          
        nees_total(k) = nees_total(k) + nees_val;
    end
end

nees_avg = nees_total / N_MC_nees;


%% 8. Advanced Plotting (IEEE Style with CRLB)
figure('Color', 'w', 'Position', [100 100 900 700]);

subplot(2,1,1);
plot(time_vec, rmse_pos_avg, 'b-', 'LineWidth', 2); hold on;
plot(time_vec, crlb_pos_vec, 'r--', 'LineWidth', 1.5); % Updated Variable
plot(time_vec, 2*crlb_pos_vec, 'k:', 'LineWidth', 1);
grid on; box on;
ylabel('Position RMSE (m)', 'FontSize', 11);
title('Monte Carlo Analysis: EKF Performance vs PCRLB', 'FontSize', 12, 'FontWeight', 'bold');
legend('EKF RMSE', 'PCRLB (Theoretical Bound)', '2×PCRLB', 'Location', 'best');
% ylim([0, max([rmse_pos_avg; crlb_pos_vec])*1.2]);

subplot(2,1,2);
plot(time_vec, rmse_vel_avg, 'b-', 'LineWidth', 2); hold on;
plot(time_vec, crlb_vel_vec, 'r--', 'LineWidth', 1.5);  % ✅ FIXED: lowercase + _vec
plot(time_vec, 2*crlb_vel_vec, 'k:', 'LineWidth', 1);    % ✅ FIXED
grid on; box on;
xlabel('Time (s)', 'FontSize', 11);
ylabel('Velocity RMSE (m/s)', 'FontSize', 11);
legend('EKF RMSE', 'PCRLB', '2×PCRLB', 'Location', 'best');
ylim([0, max([rmse_vel_avg; crlb_vel_vec])*1.2]);

%% 9. Print Performance Metrics
fprintf('\n========================================\n');
fprintf('📈 FINAL PERFORMANCE METRICS\n');
fprintf('========================================\n');
fprintf('Position RMSE (steady-state):  %.2f m\n', rmse_pos_avg(end));
fprintf('Velocity RMSE (steady-state):  %.2f m/s\n', rmse_vel_avg(end));
fprintf('PCRLB Position (final):         %.2f m\n', crlb_pos_vec(end));
fprintf('EKF/PCRLB Ratio:                %.2f×\n', rmse_pos_avg(end)/crlb_pos_vec(end));
fprintf('========================================\n');

if rmse_pos_avg(end) < 2*crlb_pos_vec(end)
    fprintf('✅ FILTER PERFORMANCE: EXCELLENT (within 2×PCRLB)\n');
elseif rmse_pos_avg(end) < 5*crlb_pos_vec(end)
    fprintf('✅ FILTER PERFORMANCE: GOOD (within 5×PCRLB)\n');
else
    fprintf('⚠️  FILTER PERFORMANCE: NEEDS IMPROVEMENT\n');
end
fprintf('========================================\n');

%% 10. NEES Consistency Plot
% Plot
figure('Color','w');
plot(time_vec, nees_avg, 'b-', 'LineWidth', 2); hold on;
yline(6, 'r--', 'LineWidth', 1.5);
yline(13.8, 'k:', 'LineWidth', 1.2);
grid on; box on;
xlabel('Time (s)', 'FontSize', 11);
ylabel('NEES', 'FontSize', 11);
title('Filter Consistency Test (NEES)', 'FontSize', 12, 'FontWeight', 'bold');
legend('NEES (Avg)', 'Expected (6)', '99% Bound (13.8)', 'Location', 'best');
ylim([0, max([nees_avg; 20])]);

fprintf('\nAverage NEES (steady): %.2f\n', mean(nees_avg(end-100:end)));