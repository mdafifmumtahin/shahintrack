function m2 = snr_to_R_mult(snr_meas_dB)
% SNR_TO_R_MULT Measured SNR (dB) theke measurement-noise multiplier.
%   >>> v3.0 core idea: R ke ML diye guess korar dorkar nei — radar
%   equation-i bole dey noise koto. CRLB onujayi sigma ~ 1/sqrt(SNR).
%   Simulator EI exact model use kore noise banay, tai deployment-e same
%   model diye R set korle measurement part optimally consistent hoy.
%   ML ekhon shudhu PROCESS noise (Q / maneuver) er upor focus kore. <<<
%
%   R_k = base_R * m2,  jekhane m2 = clip(1/sqrt(SNR_linear), 1, 10)^2

    snr_lin = max(10^(snr_meas_dB/10), 1e-4);
    m = 1 / sqrt(snr_lin);
    m = min(max(m, 1.0), 10.0);
    m2 = m^2;
end
