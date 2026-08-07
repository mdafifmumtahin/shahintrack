function cost = optimize_ekf_step(scales, temp_ekf, z_meas, x_true, dt, base_Q, base_R)
    %% 1. Apply scales to the matrices
    q_scale = scales(1);
    r_scale = scales(2);
    
    % Now base_Q is 6x6, so simple scalar multiplication works!
    temp_ekf.ProcessNoise = base_Q * q_scale; 
    temp_ekf.MeasurementNoise = base_R * r_scale;
    
    %% 2. Run predict and correct using Toolbox
    predict(temp_ekf, dt);
    correct(temp_ekf, z_meas);
    
    %% 3. Extract estimated position
    % Toolbox constvel state is [x; vx; y; vy; z; vz]
    est_pos = [temp_ekf.State(1); temp_ekf.State(3); temp_ekf.State(5)];
    true_pos = [x_true(1); x_true(2); x_true(3)];
    
    %% 4. Calculate Cost (Euclidean Distance Error)

    % 1. Position Error (RMSE component)
    rmse_cost = norm(true_pos - est_pos);
    
    %% 5. NEES Calculation (Consistency component)
    % State error vector for position (3x1)
    err_pos = true_pos - est_pos;
    
    % Extract 3x3 position covariance from 6x6 StateCovariance
    % State is ordered as [x; vx; y; vy; z; vz], so indices are 1, 3, 5
    % P_pos = temp_ekf.StateCovariance([1,3,5], [1,3,5]);
    
    % NEES formula: e = err' * inv(P) * err
    % nees_pos = err_pos' * pinv(P_pos) * err_pos;
    
    %% 6. Combined Cost Function
    % Degrees of Freedom (DOF) for 3D position is 3. Ideal NEES should be close to 3.
    % lambda_nees = 50.0; % penalty অনেক বাড়িয়ে দেওয়া হলো (আগে ৫ ছিল)
    % nees_penalty = abs(nees_pos - 3);
    
    %নন-লিনিয়ার পেনাল্টি (যাতে ৩ থেকে বেশি দূরে গেলে শাস্তি বাড়ে)
    %nees_penalty = (nees_pos - 3)^2; 

    % 2. L2 Regularization (Keep scales near 1.0 when moving straight)
    lambda_reg = 0.01; 
    scale_penalty = (q_scale - 1)^2 + (r_scale - 1)^2;
    
    cost = rmse_cost + (lambda_reg * scale_penalty);
    
    %cost = rmse_cost; % + (lambda_nees * nees_penalty)
    

end