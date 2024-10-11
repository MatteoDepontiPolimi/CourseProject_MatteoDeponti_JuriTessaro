function stop = stopTrainingIfTargetMSEReached(info, target_mse)
    stop = false;  % Default: continue training
    
    % Check if we have validation data and validation loss
    if info.State == "iteration"
        % Check if the current loss is less than the target MSE
        if info.TrainingLoss < target_mse
            stop = true; % Stop the training
            disp("Stopping training: Target MSE reached.");
        end
    end
end