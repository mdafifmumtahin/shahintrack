function [features, nis_buffer] = extract_tracker_features(res, S_cov, nis_buffer, snr_meas_dB_k)
% EXTRACT_TRACKER_FEATURES Single source of truth for the ML feature vector.
%   >>> FIX: dataset generation ar validation dutotei EI function use korbe,
%   tai feature mismatch bug ar hote parbe na. <<<
%
% Inputs:
%   res           : innovation (residual) vector [4x1] (already angle-wrapped
%                   if HasMeasurementWrapping is on)
%   S_cov         : innovation covariance [4x4]
%   nis_buffer    : rolling buffer of past NIS values (row vector)
%   snr_meas_dB_k : MEASURED SNR in dB at this step (NOT true RCS/SNR!)
%
% Output:
%   features   : [1 x 4] = [log_nis, log_avg_nis, snr_meas_dB, innov_range_norm]
%   nis_buffer : updated buffer
%
% Notes:
%  - log10 scale: NIS heavily right-skewed, log kore dile MLP-er kaj sohoj hoy.
%  - S_cov \ res  use kora hoyeche pinv-er bodole (faster + numerically better).

    nis_val = res' * (S_cov \ res);
    nis_val = max(nis_val, 1e-6);

    % Rolling window update (window size = length of buffer, e.g. 10)
    nis_buffer = [nis_val, nis_buffer(1:end-1)];
    avg_nis = mean(nis_buffer);

    % Normalized range innovation (unitless, robust extra signal)
    innov_range_norm = abs(res(1)) / sqrt(max(S_cov(1,1), 1e-9));

    features = [log10(nis_val), log10(avg_nis), snr_meas_dB_k, innov_range_norm];
end
