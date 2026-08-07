function [z_pred, bounds] = radar_meas_model(state, varargin)
% RADAR_MEAS_MODEL Shared measurement function for ALL trackers/scripts.
%   >>> FIX: ek jaygay define kora, jate training ar validation script-e
%   kokhono mismatch na hoy. <<<
%
%   Supports measurement wrapping: azimuth is periodic on [-pi, pi].
%   trackingEKF-e 'HasMeasurementWrapping', true dile innovation
%   automatically wrap hobe (2*pi jump ar NIS spike hobe na).
%
% State layout (constvel): [x; vx; y; vy; z; vz]

    px = state(1); vx = state(2);
    py = state(3); vy = state(4);
    pz = state(5); vz = state(6);

    r     = max(sqrt(px^2 + py^2 + pz^2), 1e-6);
    r_dot = (px*vx + py*vy + pz*vz) / r;
    az    = atan2(py, px);
    el    = asin(max(min(pz / r, 1), -1));

    z_pred = [r; r_dot; az; el];

    % Wrapping bounds: [-inf inf] = no wrap, [-pi pi] = periodic
    bounds = [-inf inf;    % range
              -inf inf;    % doppler
              -pi   pi;    % azimuth  <-- FIX: wrap
              -inf inf];   % elevation
end
