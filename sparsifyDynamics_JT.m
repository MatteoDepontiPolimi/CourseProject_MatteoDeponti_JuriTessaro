function Xi = sparsifyDynamics_JT(Theta,Theta1,Theta2,Theta3,dXdt,lambda,n)
    % Compute Sparse regression: sequential least squares
    Xi(:,1) = Theta1\dXdt(:,1); % Initial guess: Least-squares
    Xi(:,2) = Theta2\dXdt(:,2); % Initial guess: Least-squares
    Xi(:,3) = Theta3\dXdt(:,3); % Initial guess: Least-squares
    % Lambda is our sparsification knob.
    for k=1:10
        smallinds = (abs(Xi)<lambda); % Find small coefficients
        Xi(smallinds)=0; % and threshold
        for ind = 1:n % n is state dimension
            biginds = ~smallinds(:,ind);
            % Regress dynamics onto remaining terms to find sparse Xi
            Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind);
        end
    end
end