function [meas, meas_true] = simulate_radar_measurement(target_state, rcs, radar_params)
% SIMULATE_RADAR_MEASUREMENT Generate noisy radar measurements
%
% Input:
%   target_state : [x;y;z;vx;vy;vz] true state
%   rcs          : scalar, current RCS in m^2
%   radar_params : struct with fields:
%       .pos          : radar position [x;y;z]
%       .freq         : operating frequency (Hz)
%       .bw           : bandwidth (Hz) -> range resolution
%       .pri          : pulse repetition interval (s)
%       .sigma_range  : range measurement noise std (m)
%       .sigma_doppler: Doppler noise std (m/s)
%       .sigma_angle  : angle measurement noise std (rad)
%       .clutter_density : optional, clutter returns per resolution cell
%
% Output:
%   meas      : [range; doppler; az; el] with noise + possible clutter
%   meas_true : noise-free true measurement

    % --- Extract radar parameters ---
    rp = radar_params;
    c = 299792458;  % speed of light
    lambda = c / rp.freq;
    
    % --- True geometry ---
    rel_pos = target_state(1:3) - rp.pos;
    range_true = norm(rel_pos);
    
    % Unit vector & angles
    ux = rel_pos / range_true;
    az_true = atan2(ux(2), ux(1));
    el_true = asin(ux(3));
    
    % Radial velocity & Doppler
    rel_vel = target_state(4:6);
    radial_vel_true = dot(rel_vel, ux);  % Negative = approaching, Positive = receding
    
    % --- Measurement noise (Gaussian) ---
    range_meas = range_true + rp.sigma_range * randn;
    doppler_meas = radial_vel_true + rp.sigma_doppler * randn;
    az_meas = az_true + rp.sigma_angle * randn;
    el_meas = el_true + rp.sigma_angle * randn;
    
    % --- Optional: Simple clutter model (uniform in resolution cell) ---
    if isfield(rp, 'clutter_density') && rp.clutter_density > 0
        % Resolution cell dimensions
        delta_r = c / (2 * rp.bw);  % range resolution
        delta_az = 0.1;  % assume beamwidth, can be parameterized
        delta_el = 0.1;
        
        % Probability of clutter in this cell
        if rand < rp.clutter_density
            % Generate clutter return near true measurement
            range_meas = range_true + delta_r * (rand - 0.5);
            az_meas = az_true + delta_az * (rand - 0.5);
            el_meas = el_true + delta_el * (rand - 0.5);
            % Clutter has near-zero Doppler typically
            doppler_meas = 0 + rp.sigma_doppler * randn;
        end
    end
    
    % --- Pack outputs ---
    meas = [range_meas; doppler_meas; az_meas; el_meas];
    meas_true = [range_true; radial_vel_true; az_true; el_true];
end