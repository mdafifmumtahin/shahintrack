function [state_hist, rcs_hist, time_vec, maneuver_flag, num_maneuvers] = generate_stealth_trajectory(params)
% GENERATE_STEALTH_TRAJECTORY Generates 3D kinematic truth and Swerling-3 RCS
%
% Output:
%   state_hist : [N x 6] matrix [x, y, z, vx, vy, vz]
%   rcs_hist   : [N x 1] RCS values in linear scale (m^2)
%   time_vec   : [N x 1] Time vector (s)
    
    T = params.T; 
    dt = params.dt;
    N = floor(T/dt) + 1;
    time_vec = (0:N-1)' * dt;
    
    % Pre-allocate storage
    state_hist = zeros(N, 6);
    rcs_hist = zeros(N, 1);
    maneuver_flag = zeros(N, 1);
    
    % ---> FIX 1: Initialize the very first step! <---
    state_hist(1, 1:3) = params.init_pos(:)';
    state_hist(1, 4:6) = params.init_vel(:)';
    
    %% Swerling-3 Parameters (Chi-square with 4 DOF)
    rcs_mean_lin = 10^(params.rcs_mean_dBsm / 10); 
    scale_param = rcs_mean_lin / 4; 
    
    % ==========================================
    % MANEUVER PLANNER
    % ==========================================
    num_maneuvers = randi([1, 2]); % পুরো ফ্লাইটে ১ বা ২ বার বাঁক নেবে
    maneuver_starts = zeros(1, num_maneuvers);
    maneuver_durations = zeros(1, num_maneuvers);
    maneuver_accels = zeros(3, num_maneuvers);
    
    for m = 1:num_maneuvers
        % র‍্যান্ডম স্টার্ট টাইম (যেন একটার সাথে আরেকটা ওভারল্যাপ না করে)
        if m == 1
            maneuver_starts(m) = 5 + rand() * 15; % ৫ থেকে ২০ সেকেন্ডের মাঝে
        else
            maneuver_starts(m) = 30 + rand() * 15; % ৩০ থেকে ৪৫ সেকেন্ডের মাঝে
        end
        
        maneuver_durations(m) = 3 + rand() * 5; % ৩ থেকে ৮ সেকেন্ড ধরে টার্ন নেবে
        
        % র‍্যান্ডম 3D ডিরেকশন এবং 1G থেকে 3G ত্বরণ
        rand_dir = randn(3, 1);
        rand_dir = rand_dir / norm(rand_dir);
        rand_mag = (1.0 + rand() * 2.0) * 9.81; 
        maneuver_accels(:, m) = rand_mag * rand_dir;
    end
    
    % ==========================================
    % MAIN TRAJECTORY LOOP
    % ==========================================
    for k = 2:N
        t = time_vec(k);
        
        % চেক করো বর্তমান সময়ে কোনো ম্যানুভার অ্যাকটিভ আছে কিনা
        is_maneuvering_now = 0; % কারেন্ট স্টেপের জন্য ফ্ল্যাগ

        accel = [0; 0; 0];
        for m = 1:num_maneuvers
            if t >= maneuver_starts(m) && t <= (maneuver_starts(m) + maneuver_durations(m))
                accel = maneuver_accels(:, m);
                is_maneuvering_now = 1; % ম্যানুভার হচ্ছে!
                break;
            end
        end

        maneuver_flag(k) = is_maneuvering_now; % রেকর্ড সেভ করা হলো
        
        % Explicit Kinematics Update
        pos = state_hist(k-1, 1:3)';
        vel = state_hist(k-1, 4:6)';
        
        % ---> FIX 2: Use dt instead of t for integration! <---
        current_pos = pos + vel*dt + 0.5*accel*dt^2;
        current_vel = vel + accel*dt;
        
        state_hist(k, 1:3) = current_pos';
        state_hist(k, 4:6) = current_vel';
        
        % Swerling-3 RCS Fluctuation
        chi2_4 = sum(randn(4, 1).^2);
        rcs_hist(k) = max(scale_param * chi2_4, 1e-6); % Physical lower bound
    end
end