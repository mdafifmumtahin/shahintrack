%% SCRIPT_PLOT_TRAJECTORY
% Visualizes the Stealth Target Ground Truth and RCS Fluctuation
clear; clc; close all;

% 1. Define Parameters for a Low-Observable (Stealth) Target
params.T = 60;                        % Simulate for 60 seconds
params.dt = 0.01;                      % 10 Hz update rate
params.init_pos = [15000; 5000; 3000];% Start at 15km X, 5km Y, 3km Altitude
params.init_vel = [-250; -50; 0];     % Moving fast towards the radar
params.accel = [0; 0; 0];             % Constant Velocity model
params.rcs_mean_dBsm = -15;           % Very low RCS (-15 dBsm)

% 2. Generate Data
fprintf('Generating Stealth Target Trajectory...\n');
[state_true, rcs_true, t_vec, ~, num_maneuvers] = generate_stealth_trajectory(params);


fprintf('Number of maneuvers: %d...\n', num_maneuvers);

% 3. Visualization
figure('Color', 'w', 'Position', [100, 100, 1000, 500]);

% Plot 1: 3D Trajectory
subplot(1, 2, 1);
plot3(state_true(:,1)/1000, state_true(:,2)/1000, state_true(:,3)/1000, 'b-', 'LineWidth', 2);
hold on;
plot3(state_true(1,1)/1000, state_true(1,2)/1000, state_true(1,3)/1000, 'go', 'MarkerFaceColor', 'g'); % Start
plot3(state_true(end,1)/1000, state_true(end,2)/1000, state_true(end,3)/1000, 'rs', 'MarkerFaceColor', 'r'); % End
plot3(0, 0, 0, 'k^', 'MarkerFaceColor', 'k', 'MarkerSize', 8); % Radar Position
grid on; box on;
xlabel('X Position (km)'); ylabel('Y Position (km)'); zlabel('Altitude (km)');
title('Target 3D Trajectory');
legend('Flight Path', 'Start', 'End', 'Radar', 'Location', 'best');
view(-45, 30);

% Plot 2: Swerling-3 RCS Fluctuation over time
subplot(1, 2, 2);
rcs_dBsm_hist = 10*log10(rcs_true);
plot(t_vec, rcs_dBsm_hist, 'r-', 'LineWidth', 1.2);
hold on;
yline(params.rcs_mean_dBsm, 'k--', 'LineWidth', 1.5);
grid on; box on;
xlabel('Time (s)'); ylabel('RCS (dBsm)');
title('Swerling-3 RCS Fluctuation (Stealth Profile)');
legend('Instantaneous RCS', 'Mean RCS (-15 dBsm)', 'Location', 'best');
ylim([-40, 0]);