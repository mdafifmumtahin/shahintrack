%% ML_EKF_MONTE_CARLO_VALIDATION (v3.1)
% v3.1 tracker (physics-R + ML-Q + NIS-match) vs Static vs IMM
% Integrated with Detailed Performance and Computational Report
clear; clc; close all;

addpath('../src_simulation'); addpath('../src_common'); addpath('../models');
fprintf('Loading v3.1 Q-Oracle...\n');
load('ml_oracle_net.mat', 'ml_oracle_net');
N_MC = 20;
fprintf('Monte Carlo Validation (%d runs)...\n', N_MC);

params.T = 60; params.dt = 0.01;
params.init_pos = [12000; 4000; 2500];
params.init_vel = [-200; -40; 0];
params.rcs_mean_dBsm = -15;

radar_params.sigma_range=158; radar_params.sigma_doppler=31.6;
radar_params.sigma_angle=deg2rad(1); radar_params.sigma_snr_dB=2.0;

T=params.dt; q_var=1.0^2;
q_block=[T^4/4,T^3/2;T^3/2,T^2]*q_var;
base_Q=blkdiag(q_block,q_block,q_block);
base_R=diag([radar_params.sigma_range,radar_params.sigma_doppler, ...
             radar_params.sigma_angle,radar_params.sigma_angle].^2);

NISMATCH_GAIN=1.0; NISMATCH_CAP=1000; NISMATCH_EMA=0.1; HOLD_TAU=100;
lam_hold=exp(-1/HOLD_TAU);

N_samples=floor(params.T/params.dt)+1; window_size=10;

% --- Storage Arrays for MC Runs ---
se_static=zeros(N_samples,N_MC); 
se_ml=zeros(N_samples,N_MC);
se_imm=zeros(N_samples,N_MC);

nees_static_all=zeros(N_samples,N_MC); 
nees_ml_all=zeros(N_samples,N_MC);
nees_imm_all=zeros(N_samples,N_MC);

q_eff_log_all = zeros(N_samples,N_MC);

time_static_all = zeros(N_samples,N_MC);
time_ml_all = zeros(N_samples,N_MC);
time_imm_all = zeros(N_samples,N_MC);

for mc=1:N_MC
    fprintf('   Run %d/%d...\n', mc, N_MC);
    [state_true,rcs_true,time_vec,~,~]=generate_stealth_trajectory(params);
    [meas_noisy,~,~,snr_meas_dB]=simulate_radar_physics(state_true,rcs_true,radar_params);
    init_state=[state_true(1,1);state_true(1,4);state_true(1,2); ...
                state_true(1,5);state_true(1,3);state_true(1,6)];
    init_cov=diag([100 300 100 300 100 300].^2);  
    
    mk_ekf=@() trackingEKF('State',init_state,'StateCovariance',init_cov, ...
              'StateTransitionFcn',@constvel,'ProcessNoise',base_Q, ...
              'MeasurementFcn',@radar_meas_model,'HasMeasurementWrapping',true, ...
              'MeasurementNoise',base_R);
    
    ekf_static=mk_ekf(); 
    ekf_ml=mk_ekf();
    
    % Setup IMM
    model1 = mk_ekf(); model1.ProcessNoise = base_Q * 1.0;
    model2 = mk_ekf(); model2.ProcessNoise = base_Q * 900.0;
    imm_filter = trackingIMM('TrackingFilters', {model1, model2}, ...
                             'TransitionProbabilities', [0.95 0.05; 0.05 0.95]);

    nis_buffer=ones(1,window_size)*4;
    q_ml=1.0; q_floor=1.0; q_boost=1.0; nis_ema=4.0;
    
    for k=2:N_samples
        z=meas_noisy(k,:)'; true_pos=state_true(k,1:3)';
        
        % --- STATIC ---
        tic; predict(ekf_static,params.dt); correct(ekf_static,z); time_static_all(k,mc) = toc;
        e=true_pos-ekf_static.State([1,3,5]); se_static(k,mc)=sum(e.^2);
        P=ekf_static.StateCovariance([1,3,5],[1,3,5]); nees_static_all(k,mc)=e'*(P\e);
        
        % --- ML v3.1 ---
        tic;
        ekf_ml.MeasurementNoise=base_R*snr_to_R_mult(snr_meas_dB(k));
        ekf_ml.ProcessNoise=base_Q*q_ml*q_boost;
        predict(ekf_ml,params.dt);
        [res,S_cov]=residual(ekf_ml,z);
        [features,nis_buffer]=extract_tracker_features(res,S_cov,nis_buffer,snr_meas_dB(k));
        q_raw=double(10^predict(ml_oracle_net,features));
        nis_now=res'*(S_cov\res);
        if nis_now>chi2inv(0.99,4), q_raw=max(q_raw,300); end   
        if q_raw>q_ml, q_ml=0.1*q_ml+0.9*q_raw; else, q_ml=0.85*q_ml+0.15*q_raw; end
        q_floor=max(q_ml,q_floor*lam_hold); q_ml=min(max(max(q_ml,q_floor),0.1),1000);
        nis_val=res'*(S_cov\res);
        nis_ema=(1-NISMATCH_EMA)*nis_ema+NISMATCH_EMA*nis_val;
        q_boost=min(max(q_boost*(nis_ema/4.0)^NISMATCH_GAIN,1/NISMATCH_CAP),NISMATCH_CAP);
        correct(ekf_ml,z);
        time_ml_all(k,mc) = toc;
        q_eff_log_all(k,mc) = q_ml * q_boost;
        e=true_pos-ekf_ml.State([1,3,5]); se_ml(k,mc)=sum(e.^2);
        P=ekf_ml.StateCovariance([1,3,5],[1,3,5]); nees_ml_all(k,mc)=e'*(P\e);
        
        % --- IMM ---
        tic; predict(imm_filter, params.dt); correct(imm_filter, z); time_imm_all(k,mc) = toc;
        e=true_pos-imm_filter.State([1,3,5]); se_imm(k,mc)=sum(e.^2);
        P=imm_filter.StateCovariance([1,3,5],[1,3,5]); nees_imm_all(k,mc)=e'*(P\e);
    end
end

% --- Process MC Metrics ---
rmse_t_static = sqrt(mean(se_static,2)); 
rmse_t_ml = sqrt(mean(se_ml,2));
rmse_t_imm = sqrt(mean(se_imm,2));
avg_q_eff_log = mean(q_eff_log_all, 2);

mean_rmse_static = sqrt(mean(se_static(2:end,:),'all'));
mean_rmse_ml = sqrt(mean(se_ml(2:end,:),'all'));
mean_rmse_imm = sqrt(mean(se_imm(2:end,:),'all'));

improvement_static = (mean_rmse_static - mean_rmse_ml) / mean_rmse_static * 100;
improvement_imm = (mean_rmse_imm - mean_rmse_ml) / mean_rmse_imm * 100;

avg_time_static = mean(time_static_all(2:end,:), 'all') * 1000;
max_time_static = max(time_static_all(2:end,:), [], 'all') * 1000;
avg_time_ml = mean(time_ml_all(2:end,:), 'all') * 1000;
max_time_ml = max(time_ml_all(2:end,:), [], 'all') * 1000;
avg_time_imm = mean(time_imm_all(2:end,:), 'all') * 1000;
max_time_imm = max(time_imm_all(2:end,:), [], 'all') * 1000;
overhead = avg_time_ml - avg_time_static;

med_nees_static = median(nees_static_all(2:end,:), 'all');
med_nees_ml = median(nees_ml_all(2:end,:), 'all');
med_nees_imm = median(nees_imm_all(2:end,:), 'all');

% --- PRINT RESULTS ---
fprintf('\n📊 --- MONTE CARLO RESULTS (%d runs) --- 📊\n', N_MC);
fprintf('Static EKF Avg Position RMSE: %.2f meters\n', mean_rmse_static);
fprintf('ML-EKF Avg Position RMSE:     %.2f meters\n', mean_rmse_ml);
fprintf('IMM Filter Avg Position RMSE: %.2f meters\n', mean_rmse_imm);
fprintf('-------------------------------------------\n');
fprintf('Performance Improvement (ML vs Static):  %.2f%%\n', improvement_static);
fprintf('Performance Gap (ML vs IMM Optimal):     %.2f%% (positive = ML better)\n', improvement_imm);
fprintf('-------------------------------------------\n');

fprintf('\n📊 --- COMPUTATIONAL COMPLEXITY REPORT --- 📊\n');
fprintf('Target Deadline (100Hz): 10.0000 ms\n\n');
fprintf('Static EKF:\n');
fprintf('  Average Processing Time : %.4f ms\n', avg_time_static);
fprintf('  Max Processing Time     : %.4f ms\n\n', max_time_static);
fprintf('ML-EKF:\n');
fprintf('  Average Processing Time : %.4f ms\n', avg_time_ml);
fprintf('  Max Processing Time     : %.4f ms\n\n', max_time_ml);
fprintf('IMM-EKF:\n');
fprintf('  Average Processing Time : %.4f ms\n', avg_time_imm);
fprintf('  Max Processing Time     : %.4f ms\n', max_time_imm);
fprintf('\nML Oracle Overhead (MATLAB) : %.4f ms\n', overhead);
fprintf('-------------------------------------------\n');

fprintf('\n📊 --- PERFORMANCE METRICS --- 📊\n');
fprintf('Static EKF Median NEES : %.2f (Ideal: ~3.0)\n', med_nees_static);
fprintf('ML-EKF Median NEES     : %.2f (Ideal: ~3.0)\n', med_nees_ml);
fprintf('IMM-EKF Median NEES    : %.2f (Ideal: ~3.0)\n', med_nees_imm);
fprintf('---------------------------------\n');

%% 7. Plotting the Results
figure('Color', 'w', 'Position', [100, 100, 1000, 600]);

% Subplot 1: Position Error Comparison
subplot(2, 2, 1);
plot(time_vec, rmse_t_static, 'b-', 'LineWidth', 1.5); hold on;
plot(time_vec, rmse_t_ml, 'r-', 'LineWidth', 2);
plot(time_vec, rmse_t_imm, 'm-.', 'LineWidth', 1.5);
title(sprintf('MC-Averaged Position Error (%d runs)', N_MC));
xlabel('Time (s)'); ylabel('RMSE (meters)');
legend('Static EKF', 'ML-EKF', 'IMM Filter', 'Location', 'best');
grid on; ylim([0 400]);

% Subplot 2: Q Scaling
subplot(2,2,2);
semilogy(time_vec, max(avg_q_eff_log, 0.1), 'r-', 'LineWidth', 1.5);
title('Average ML Adaptive Q Scaling Factor');
xlabel('Time (s)'); ylabel('Q Scale Factor (Log)');
grid on;

% Subplot 3: Zoomed Error Comparison
subplot(2, 2, 3);
plot(time_vec, rmse_t_static, 'b-', 'LineWidth', 1.2); hold on;
plot(time_vec, rmse_t_ml, 'r-', 'LineWidth', 1.5);
plot(time_vec, rmse_t_imm, 'm-.', 'LineWidth', 1.5);
grid on; box on;
xlabel('Time (s)'); ylabel('RMSE (m)');
title('Tracking Error Comparison (Zoomed)');
legend('Static EKF', 'ML-EKF', 'IMM Filter', 'Location', 'best');
ylim([0, max(mean_rmse_static)*2]); 

% Subplot 4: NEES Consistency
subplot(2, 2, 4);
plot(time_vec, median(nees_static_all, 2), 'b-', 'LineWidth', 1.2); hold on;
plot(time_vec, median(nees_ml_all, 2), 'r-', 'LineWidth', 1.5);
plot(time_vec, median(nees_imm_all, 2), 'm-.', 'LineWidth', 1.2);
yline(3, 'k--', 'LineWidth', 2); 
grid on; box on;
xlabel('Time (s)'); ylabel('Median NEES');
title('MC Median Filter Consistency (NEES)');
legend('Static EKF', 'ML-EKF', 'IMM Filter', 'Ideal Bound', 'Location', 'best');
ylim([0, 15]);