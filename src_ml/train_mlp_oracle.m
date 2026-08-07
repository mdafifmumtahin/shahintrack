%% TRAIN_MLP_ORACLE (v3.0)
% Single-output Q oracle. Run-level 80-20 split, oversample maneuver
% samples in TRAIN only. (v2.x-er sob leakage-fix bohal ache.)
clear; clc; close all;

fprintf('Loading Training Data...\n');
load('../data/ml_training_set.mat');   % X_train, Y_train (Nx1), run_id

numFeatures = size(X_train, 2);   % 4

% Run-level split
runs = unique(run_id); rng(42);
shuffled = runs(randperm(numel(runs)));
n_val = max(1, round(0.2*numel(runs)));
val_runs = shuffled(1:n_val); train_runs = shuffled(n_val+1:end);
idxTr = ismember(run_id, train_runs); idxVa = ismember(run_id, val_runs);
X_tr = X_train(idxTr,:); Y_tr = Y_train(idxTr,:);
X_va = X_train(idxVa,:); Y_va = Y_train(idxVa,:);
fprintf('Split: %d train runs, %d val runs\n', numel(train_runs), n_val);

% Oversample maneuver samples (TRAIN only)
man = find(Y_tr > 0.5*log10(900));
num_copies = 4;
X_tr = [X_tr; repmat(X_tr(man,:), num_copies, 1)];
Y_tr = [Y_tr; repmat(Y_tr(man,:), num_copies, 1)];
fprintf('Oversampled %d maneuver samples x%d\n', numel(man), num_copies);

layers = [
    featureInputLayer(numFeatures, 'Normalization', 'zscore', 'Name', 'input')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.1, 'Name', 'drop1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'output')   % single output
    regressionLayer('Name', 'reg')
];
options = trainingOptions('adam', 'MaxEpochs', 60, 'MiniBatchSize', 1024, ...
    'InitialLearnRate', 0.005, 'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, 'LearnRateDropPeriod', 25, ...
    'ValidationData', {X_va, Y_va}, 'ValidationFrequency', 50, ...
    'ValidationPatience', 8, 'Shuffle', 'every-epoch', ...
    'Plots', 'none', 'Verbose', true);

fprintf('Training v3.0 Q-oracle...\n');
[ml_oracle_net, info] = trainNetwork(X_tr, Y_tr, layers, options);
if ~exist('../models','dir'), mkdir('../models'); end
save('../models/ml_oracle_net.mat', 'ml_oracle_net');
fprintf('Model saved (val RMSE %.4f)\n', info.FinalValidationRMSE);
