%% RUN_ALL (v3.0) — Ek command-e puro pipeline
% Project ROOT-e ei file. MATLAB-e open kore Run (F5) chapo.
this_file = mfilename('fullpath');
project_root = fileparts(this_file);
required = {'src_ml','src_simulation','src_common'};
for i = 1:numel(required)
    if ~isfolder(fullfile(project_root, required{i}))
        error('Folder "%s" nei %s-e. Zip abar extract koro.', required{i}, project_root);
    end
end
if ~isfolder(fullfile(project_root,'data')),   mkdir(fullfile(project_root,'data'));   end
if ~isfolder(fullfile(project_root,'models')), mkdir(fullfile(project_root,'models')); end
cd(fullfile(project_root,'src_ml'));

fprintf('==== STEP 1/4: Dataset Generation ====\n'); generate_optimized_dataset;
fprintf('\n==== STEP 2/4: Training Q-Oracle ====\n'); train_mlp_oracle;
fprintf('\n==== STEP 3/4: Single-Run Validation ====\n'); run_ml_ekf_validation;
fprintf('\n==== STEP 4/4: Monte Carlo Validation ====\n'); ML_EKF_monte_carlo_validation;
fprintf('\nSOB SHESH! Plots ar RESULTS table dekho.\n');
