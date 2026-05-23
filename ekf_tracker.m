function [est_state, est_cov] = ekf_tracker(meas_hist, init_state, init_cov, q_std, r_std_vec, dt)
% EKF_TRACKER Extended Kalman Filter for Radar Tracking
%
% Input:
%   meas_hist : [N x 4] matrix [range, radial_vel, az, el]
%   init_state: [6x1] initial state [x;y;z;vx;vy;vz]
%   init_cov  : [6x6] initial covariance
%   q_std     : Process noise std for velocity (m/s^2)
%   r_std_vec : [4x1] Measurement noise std [sigma_r, sigma_v, sigma_az, sigma_el]
%   dt        : Time step
%
% Output:
%   est_state : [N x 6] Estimated state history
%   est_cov   : [N x 6x6] Covariance history (optional, can be last one)

    N = size(meas_hist, 1);
    est_state = zeros(N, 6);
est_cov = zeros(N, 6, 6);  % ✅Pre-allocate 3D covariance array
    P = init_cov;
    x = init_state;
    
    % Process Noise Covariance (Discrete White Noise Acceleration)
    % Q matrix for 6-state CV model
    q = q_std^2;
    T = dt;
    G = [T^2/2 0 0; 0 T^2/2 0; 0 0 T^2/2; T 0 0; 0 T 0; 0 0 T];
    Q = G * diag([q q q]) * G';
    
    % Measurement Noise Covariance
    R = diag(r_std_vec.^2);
    
        for k = 1:N
        %% 1. Prediction Step ===
        F = [1 0 0 T 0 0;
             0 1 0 0 T 0;
             0 0 1 0 0 T;
             0 0 0 1 0 0;
             0 0 0 0 1 0;
             0 0 0 0 0 1];
             
        x_pred = F * x;
        P_pred = F * P * F' + Q;
        P_pred = (P_pred + P_pred') / 2;  %  FIX 1: Force Symmetry
        
        %% 2. Measurement Prediction & Jacobian ===
        px = x_pred(1); py = x_pred(2); pz = x_pred(3);
        vx = x_pred(4); vy = x_pred(5); vz = x_pred(6);
        
        % 🔒 FIX 2: Geometric Safeguards (Avoid division by zero)
        r = sqrt(px^2 + py^2 + pz^2);
        r = max(r, 1e-6);
        rho2 = px^2 + py^2;
        rho2 = max(rho2, 1e-6);
        
        r_dot = (px*vx + py*vy + pz*vz) / r;
        az = atan2(py, px);
        el = asin(pz / r);
        z_pred = [r; r_dot; az; el];
        
        % Jacobian H
        H = zeros(4, 6);
        H(1,1:3) = [px, py, pz] / r;
        H(2,1:3) = [(vx*r^2 - px*r*r_dot)/r^3, ...
                    (vy*r^2 - py*r*r_dot)/r^3, ...
                    (vz*r^2 - pz*r*r_dot)/r^3];
        H(2,4:6) = [px, py, pz] / r;
        H(3,1) = -py / rho2;  H(3,2) =  px / rho2;
        
        sin_el = pz / r;
        cos_el = sqrt(max(0, 1 - sin_el^2));
        H(4,1:3) = [-px*sin_el, -py*sin_el, r*cos_el] / r^2;
        
        %% 3. Update Step with Gating ===
        z = meas_hist(k, :)';
        
        % Innovation
        nu = z - z_pred;
        nu(3) = wrapToPi(nu(3));
        
        % Innovation Covariance
        S = H * P_pred * H' + R;
        S = (S + S') / 2;
        S = S + eye(4) * 1e-9;
        
        % Mahalanobis Distance (Gating)
        gamma = nu' * (S \ nu);
        
        % ✅ TIGHT GATING: Reject outliers aggressively
        gate_threshold = 9.21;  % 95% confidence for 4 DOF (chi2inv(0.95,4))
        
        if gamma < gate_threshold
            % VALID measurement: Update with Kalman Gain
            K = (P_pred * H') / S;
            x = x_pred + K * nu;
            
            % Joseph Form Covariance Update
            I_KH = eye(6) - K * H;
            P = I_KH * P_pred * I_KH' + K * R * K';
            P = (P + P') / 2;
        else
            % ❌ REJECTED: Clutter detected - SKIP UPDATE
            % Keep prediction only (coast mode)
            x = x_pred;
            P = P_pred;
            
            % Optional: Increase process noise during coast to reflect uncertainty
            P(4:6, 4:6) = P(4:6, 4:6) + eye(3) * (q_std^2 * T^2);
        end
        
        est_state(k, :) = x';
        est_cov(k, :, :) = P;
    end
end