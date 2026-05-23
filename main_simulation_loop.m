%% === Radar Simulation Main Loop Template ===
clear; clc; close all;

%% 1. Simulation parameters
sim_params.T = 20;          % total time (s)
sim_params.dt = 0.01;       % time step (s)
N = floor(sim_params.T/sim_params.dt) + 1;

%% 2. Target parameters (stealth-like)
target_params.T = 20;                  % <-- ADD THIS
target_params.dt = 0.01;               % <-- ADD THIS
target_params.init_pos = [10e3; 0; 5e3];      % 10 km range, 5 km altitude
target_params.init_vel = [-200; 0; 0];         % approaching at 200 m/s
target_params.accel = [0; 0; 0];               % non-maneuvering
target_params.rcs_mean_dBsm = -10;             % -10 dBsm = 0.1 m^2 (low observable)
target_params.swerling_type = 3;

%% 3. Radar parameters (X-band example)
radar_params.pos = [0; 0; 0];                  % ground-based
radar_params.freq = 10e9;                      % 10 GHz
radar_params.bw = 10e6;                        % 10 MHz -> 15m range res
radar_params.pri = 1e-3;                       % 1 kHz PRF
radar_params.sigma_range = 10;                 % 10m range noise
radar_params.sigma_doppler = 2;                % 2 m/s Doppler noise
radar_params.sigma_angle = 0.5*deg2rad(1);     % 1 deg angle noise
radar_params.clutter_density = 0.05;           % 5% chance of clutter per cell

%% 4. Pre-allocate storage (CRITICAL for large N)
N = floor(target_params.T/target_params.dt) + 1;  % <-- Use target_params
time_vec = zeros(N,1);
true_state = zeros(N,6);
true_rcs = zeros(N,1);
meas_data = zeros(N,4);      % [range, doppler, az, el]
meas_true = zeros(N,4);

%% 5. Generate target truth
[state_hist, rcs_hist] = generate_point_target(target_params);

%% 6. Main measurement loop
for k = 1:N
    % Current truth
    curr_state = state_hist(k,:)';
    curr_rcs = rcs_hist(k);
    
    % Generate measurement
    [meas, meas_t] = simulate_radar_measurement(curr_state, curr_rcs, radar_params);
    
    % Store
    time_vec(k) = (k-1)*sim_params.dt;
    true_state(k,:) = curr_state';
    true_rcs(k) = curr_rcs;
    meas_data(k,:) = meas';
    meas_true(k,:) = meas_t';
end

%% 7. Run EKF Tracker
% Initial guess (can be from first measurement converted to Cartesian)
% For simplicity, let's use True State + noise for init
init_x = true_state(1,:) + randn(1,6).*[100 100 100 10 10 10]; 
init_P = diag([1000^2, 1000^2, 1000^2, 50^2, 50^2, 50^2]);

% Filter Parameters
ekf_q_std = 1.0;       % Process noise accel
ekf_r_std = [radar_params.sigma_range, radar_params.sigma_doppler, ...
             radar_params.sigma_angle, radar_params.sigma_angle];

[est_states, ~] = ekf_tracker(meas_data, init_x', init_P, ekf_q_std, ekf_r_std, sim_params.dt);


%% 8. Quick visualization (optional)
figure;
subplot(2,2,1); plot(time_vec, meas_data(:,1), 'r.', 'MarkerSize',1); hold on;
plot(time_vec, meas_true(:,1), 'b-', 'LineWidth',1.5);
ylabel('Range (m)'); legend('Noisy','True'); grid on;

subplot(2,2,2); plot(time_vec, meas_data(:,2), 'r.', 'MarkerSize',1); hold on;
plot(time_vec, meas_true(:,2), 'b-');
ylabel('Doppler (m/s)'); grid on;

subplot(2,2,3); plot(time_vec, rad2deg(meas_data(:,3)), 'r.', 'MarkerSize',1);
ylabel('Azimuth (deg)'); grid on;

subplot(2,2,4); semilogy(time_vec, true_rcs, 'k-');
ylabel('RCS (m^2)'); xlabel('Time (s)'); grid on;
title('Swerling-3 RCS Fluctuation');

%% 9. Plot Tracking Results
figure;
subplot(3,1,1);
plot(time_vec, true_state(:,1), 'b', time_vec, est_states(:,1), 'r--');
ylabel('X pos (m)'); legend('True', 'EKF'); grid on;

subplot(3,1,2);
plot(time_vec, true_state(:,4), 'b', time_vec, est_states(:,4), 'r--');
ylabel('Vx (m/s)'); grid on;

subplot(3,1,3);
plot(time_vec, meas_data(:,1), '.', time_vec, true_state(:,1), 'b', time_vec, est_states(:,1), 'r--');
ylabel('Range'); legend('Meas', 'True', 'EKF'); grid on;
