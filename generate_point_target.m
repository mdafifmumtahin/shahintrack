function [state_hist, rcs_hist] = generate_point_target(params)
% GENERATE_POINT_TARGET Generate target trajectory & Swerling-3 RCS
%
% Input:
%   params - struct with fields:
%       .T          : total simulation time (s)
%       .dt         : time step (s)
%       .init_pos   : [x;y;z] initial position (m)
%       .init_vel   : [vx;vy;vz] initial velocity (m/s)
%       .accel      : [ax;ay;az] constant acceleration (m/s^2), optional
%       .rcs_mean_dBsm : mean RCS in dBsm (e.g., -10 for stealth)
%       .swerling_type : currently supports 3
%
% Output:
%   state_hist : [N x 6] matrix [x y z vx vy vz]
%   rcs_hist   : [N x 1] RCS values in linear scale (m^2)

    % --- Parameter parsing ---
    T = params.T; dt = params.dt;
    N = floor(T/dt) + 1;
    init_pos = params.init_pos(:);
    init_vel = params.init_vel(:);
    accel = params.accel(:);
    rcs_mean_dBsm = params.rcs_mean_dBsm;
    
    % --- Pre-allocation (Critical for performance) ---
    state_hist = zeros(N, 6);
    rcs_hist = zeros(N, 1);
    
    % --- Initialize ---
    state_hist(1, 1:3) = init_pos';
    state_hist(1, 4:6) = init_vel';
    
    % Swerling-3 parameters
    rcs_mean_lin = 10^(rcs_mean_dBsm/10);  % dBsm -> linear (m^2)
    % Swerling-3: Chi-square with 4 DOF, scaled to have mean = rcs_mean_lin
    % PDF: p(x) = 4x/mean^2 * exp(-2x/mean)
    scale_param = rcs_mean_lin / 2;  % because mean of chi2(4) = 4, so 4*scale = mean => scale = mean/4? 
    % Correction: For Swerling-3, RCS = (mean/4) * chi2(4)
    scale_param = rcs_mean_lin / 4;
    
    % --- Main trajectory loop ---
    for k = 2:N
        t = (k-1)*dt;
        
        % Constant acceleration model (can be extended to maneuvering)
        pos = init_pos + init_vel*t + 0.5*accel*t^2;
        vel = init_vel + accel*t;
        
        state_hist(k, 1:3) = pos';
        state_hist(k, 4:6) = vel';
        
        % --- Swerling-3 RCS fluctuation (scan-to-scan) ---
        % Generate chi-square with 4 DOF: sum of squares of 4 N(0,1)
        chi2_4 = sum(randn(4,1).^2);
        rcs_hist(k) = scale_param * chi2_4;
        
        % Ensure RCS is physically meaningful
        rcs_hist(k) = max(rcs_hist(k), 1e-6);
    end
end