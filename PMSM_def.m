%% START FROM HERE FOR: SINDy
% clc;
% clear;
% close all;
% 
% %% Motor parameters:
% pp = 5; % Number of pole pairs of the motor
% 
% Ld = 4.4*10^-3; % d-axis inductance in Henries (H)
% Lq = 4.4*10^-3; % q-axis inductance in Henries (H)
% Rs = 0.201;     % Stator resistance in Ohms (Ω)
% phi = sqrt(2)*0.0805; % Magnetic flux linkage (in Vs, Volts*seconds)
% 
% J = 9.8*10^-3;  % Rotor inertia (moment of inertia) in kg·m²
% Vn = 190;       % Nominal voltage in Volts (V)
% In = 10.2;      % Nominal current in Amps (A)
% wm_n = 2*pi*4500/60;  % Nominal mechanical speed in rad/s (converted from 4500 RPM)
% 
% % Current controller parameters for d-axis and q-axis with fc_I = 1000 Hz (Current
% % closed-loop cut-off frequency)
% KI_d = 29087;  % Integral gain for d-axis current controller
% Kp_d = 32.14;  % Proportional gain for d-axis current controller
% KI_q = 29087;  % Integral gain for q-axis current controller
% Kp_q = 32.14;  % Proportional gain for q-axis current controller
% 
% % Uncomment this section if you are using fc_I = 500 Hz 
% % % Current controller parameters for d-axis and q-axis with fc_I = 500 Hz
% % KI_d = 7.5701e3;  % Integral gain for d-axis current controller
% % Kp_d = 16.0562;   % Proportional gain for d-axis current controller
% % KI_q = 7.5701e3;  % Integral gain for q-axis current controller
% % Kp_q = 16.0562;   % Proportional gain for q-axis current controller
% 
% KI_w = 13.15;     % Integral gain for speed controller
% Kp_w = 0.5777;    % Proportional gain for speed controller
% 
% % P.U. (Per Unit) base parameters
% Vb = sqrt(2)*Vn;   % Base voltage in the Per Unit system
% Ib = sqrt(2)*In;   % Base current in the Per Unit system
% wb = pp*wm_n;      % Base angular speed in rad/s (electrical speed)
% 
% Zb = Vb/Ib;        % Base impedance in Ohms (Ω)
% Lb = Zb/wb;        % Base inductance in Henries (H)
% phib = Vb/wb;      % Base magnetic flux in Vs (Volt-seconds)
% 
% rs = Rs/Zb;        % Per-unit stator resistance
% ld = Ld/Lb;        % Per-unit d-axis inductance
% lq = Lq/Lb;        % Per-unit q-axis inductance
% phi_pu = phi/phib; % Per-unit magnetic flux linkage
% 
% Tb = 3/2 * pp * phib * Ib;  % Base torque in Nm (Newton-meters)
% Hm = 0.5 * (J * wm_n^2) / (Tb * wm_n); % Normalized rotor inertia
% 
% % Reference values (to be adjusted for NN tuning)
% wm_slope = 0.025; % Reference mechanical speed slope (rad/s per time unit)
% id_ref = 3;       % Reference d-axis current
% Tl_cost = 1;      % Control load torque constant parameter
% k_friction = (0.05 * Tb) / (wm_n^2); % Friction torque, 5% of nominal torque at rated speed
% 
% % Noise parameters: set to zero for clean measurements
% sig_i = (0.05/Ib)^2;  % (0.05/Ib)^2 Variance of current measurement noise (P.U.)
% sig_w = (1/wm_n)^2;   % (1/wm_n)^2 Variance of speed measurement noise (P.U.)
% 
% % Simulation step size and filter settings
% t_step = 1e-4;        % Sample time in seconds
% f_filter = 5000;      % Filter cutoff frequency in Hz
% 
% % Turn on/off filter flag (1 = on, 0 = off)
% filter_on = 0;
% 
% %% PARAMETRI SIMULAZIONE
% modello='PMSM_sim.slx';
% 
% % Set simulation
% StartTime=0;
% FinishTime=15;
% Max_step_size=1e-3;
% Relative_tolerance=1e-5;
% 
% simOut=sim(modello,'StartTime',num2str(StartTime),'Stoptime',num2str(FinishTime));
% 
% 
% %% SIMULAZIONE SIMULINK
% 
% data=simOut.get('simout');
% vd=data(:,1);
% vq=data(:,2);
% id=data(:,3);
% iq=data(:,4);
% wm=data(:,5);
% Tl=data(:,6);
% Ce=data(:,7);
% 
% X_U = [id, iq, wm, vd, vq, Tl];
% t = (StartTime:t_step:FinishTime)';
% 
% %% PCA filtering
% % 
% % Dataset = [id,iq,wm,vd,vq,Tl./Tb];
% % 
% % % Create mean dataset and subtract it from the original one
% % Dataset_mean = mean(Dataset,1);
% % B = Dataset - ones(size(Dataset)).*Dataset_mean;
% % B_T = B';
% % % Calculate the covariance matrix: this tells about the link between each
% % % measurement. The idea of PCA is to project the data on a new coordinate
% % % system in which the covariances are zero (no correlations)
% % C= 1/(size(B,1)-1).*B_T*B;
% % 
% % % Calculate PCA
% % [U,S,V] = svd(B,'econ');
% % 
% % plot(t,U(:,1))
% % hold on
% % plot(t,U(:,2))
% % plot(t,U(:,3))
% % plot(t,U(:,4))
% % plot(t,U(:,5))
% % plot(t,U(:,6))
% % legend U1 U2 U3 U4 U5 U6
% % 
% % % Variances in the system: the largest ones represent the main dynamics of
% % % the system
% % figure(2)
% % stem(1:size(S,1),diag(S)./sum(S,"all")*100,'LineWidth',2)
% % yline(1,'LineWidth',3,'Color',"r")
% % ylabel('%')
% % title('Percent variances','interpreter','latex')
% % set(gca,'FontSize',20)
% % axis([0 7 0 100])
% % grid on
% % %% Data reconstruction
% % 
% % % Only variances above 1% are kept (r <= 3 by inspection)
% % r = 3; 
% % U_r = zeros(size(U,1),r); 
% % U_r_T = U_r; 
% % 
% % U_r = U(:,1:r);
% % V_T = V';
% % V_r_T = V_T(1:r,:);
% % S_r = S(1:r,1:r);
% % 
% % %%
% % clear Dataset_filter
% % tic
% % % Filter = zeros(size(U_r_T,2));
% % Dataset_filter = zeros(size(Dataset));
% % for i = 1:size(U_r,1)
% %     Filter_row = U_r(i,:)*S_r;
% %     Dataset_filter(i,:) = Filter_row*V_r_T;
% % end
% % toc
% % 
% % Dataset_filter = Dataset_filter + Dataset_mean;
% % %% Test Database
% % close all
% % 
% % for i = 1:size(Dataset_filter,2)
% %     figure(i)
% %     plot(t,Dataset(:,i),"blue")
% %     hold on
% %     plot(t,Dataset_filter(:,i),"red")
% % end
% % 
% % clear U_r U_r_T Filter
% % grid on
% % legend X X^*
% % 
% % %% Filtered data
% % id = double(Dataset_filter(:,1));
% % iq = double(Dataset_filter(:,2));
% % wm = double(Dataset_filter(:,3));
% % % vd = double(Dataset_filter(:,4));
% % % vq = double(Dataset_filter(:,5));
% % % Cm = double(Tb.*Dataset_filter(:,6));
% % 
% % X_U = [id, iq, wm, vd, vq, Tl];
% % 
% % close all
% 
% %% SINDYc
% n=3;
% 
% % Derivata numerica
% d_id = gradient(id, t);
% d_iq = gradient(iq, t);
% d_wm = gradient(wm, t);
% 
% x=[id iq wm];
% dx=[d_id d_iq d_wm];
% 
% lambda = 0.1; % Parametro di sparsificazione
% 
% % SINDy
% Theta = [x(:,1) x(:,2) x(:,3)  ... % x(:,1).*x(:,1) x(:,2).*x(:,2) x(:,3).*x(:,3)
%          x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) vd vq Tl ]; %  Costruzione della libreria di funzioni
% Xi = sparsifyDynamics(Theta, dx, lambda, n); % Regressione sparsa
% 
% % % Modified SINDy
% % Theta = [x(:,1) x(:,2) x(:,3) ... %
% %          x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) vd vq Tl ]; % Costruzione della libreria di funzioni
% % 
% % Theta1 = [x(:,1) 0*x(:,2) 0*x(:,3) ... %
% %          x(:,1).*x(:,2) 0*x(:,1).*x(:,3) x(:,2).*x(:,3) vd 0*vq 0*Tl ]; % Costruzione della libreria di funzioni
% % 
% % Theta2 = [0*x(:,1) x(:,2) x(:,3) ... %
% %          x(:,1).*x(:,2) x(:,1).*x(:,3) 0*x(:,2).*x(:,3) 0*vd vq 0*Tl ]; % Costruzione della libreria di funzioni
% % 
% % Theta3 = [0*x(:,1) x(:,2) x(:,3) ... %
% %          x(:,1).*x(:,2) 0*x(:,1).*x(:,3) 0*x(:,2).*x(:,3) 0*vd 0*vq Tl ]; % Costruzione della libreria di funzioni
% % 
% % Xi = sparsifyDynamics_JT(Theta,Theta1,Theta2,Theta3, dx, lambda, n); % Regressione sparsa
% 
% %%
% legends = {'$i_{sd}$','$i_{sq}$','$\omega_{m}$','$v_{sd}$','$v_{sq}$','$T_{l}$'};
% color = ["red","black","green","blue","cyan","magenta"];
% fontsize = 30;
% figure(1)
% tiledlayout(3,2)
% 
% for i = 1:2
%     for j = 1:3
%         nexttile(i + 2*(j-1))
%         plot(t, X_U(:,j + 3*(i-1)), color(j + 3*(i-1)), 'LineWidth', 1); % id in blu
%         if min(X_U(:,j + 3*(i-1))) > 0
%             axis([0 15 0.75*min(X_U(:,j + 3*(i-1))) 1.25*max(X_U(:,j + 3*(i-1)))])
%         else
%             axis([0 15 1.25*min(X_U(:,j + 3*(i-1))) 1.25*max(X_U(:,j + 3*(i-1)))])
%         end
%         grid on
%         legend(legends(j + 3*(i-1)),'Interpreter','latex','FontSize',fontsize)
%         ax = gca;
%         ax.FontSize = fontsize;
%     end
% end
% %% Zoom on Noise
% close all
% 
% legends = {'$i_{sd}$','$i_{sq}$','$\omega_{m}$'};
% color = ["red","black","green"];
% fontsize = 30;
% figure(1)
% tiledlayout(3,1)
% 
% i = 1;
% t_zoom = 8/t_step; tend_zoom = 9/t_step;
% for j = 1:3
%     nexttile
%     plot(t, X_U(:,j + 3*(i-1)), color(j ), 'LineWidth', 1,'LineStyle','-'); % id in blu
%     if min(X_U(t_zoom:tend_zoom,j + 3*(i-1))) > 0
%         axis([t_zoom*t_step tend_zoom*t_step 0.95*min(X_U(t_zoom:tend_zoom,j + 3*(i-1))) 1.05*max(X_U(t_zoom:tend_zoom,j + 3*(i-1)))])
%     else
%         axis([t_zoom*t_step tend_zoom*t_step 1.05*min(X_U(t_zoom:tend_zoom,j + 3*(i-1))) 1.05*max(X_U(t_zoom:tend_zoom,j + 3*(i-1)))])
%     end
%     grid on
%     legend(legends(j),'Interpreter','latex','FontSize',fontsize)
%     ax = gca;
%     ax.FontSize = fontsize;
% end
% 
% %% Zoom on Noise + PCA reconstruction
% % close all
% % 
% % legends = {'$i_{sd}$','$i_{sq}$','$\omega_{m}$','$\tilde{i}_{sd}$','$\tilde{i}_{sq}$','$\tilde{\omega}_m$'};
% % color = ["blue","cyan","magenta","red","black","green"];
% % fontsize = 30;
% % figure(1)
% % tiledlayout(3,1)
% % 
% % i = 1;
% % t_zoom = 8/t_step; tend_zoom = 8.25/t_step;
% % for j = 1:3
% %     nexttile
% %     plot(t,data(:,j+2),color(j + 3), 'LineWidth', 1)
% %     hold on
% %     plot(t, X_U(:,j + 3*(i-1)), color(j + 3*(i-1)), 'LineWidth', 1,'LineStyle','--'); % id in blu
% %     if min(X_U(t_zoom:tend_zoom,j + 3*(i-1))) > 0
% %         axis([t_zoom*t_step tend_zoom*t_step 0.95*min(X_U(t_zoom:tend_zoom,j + 3*(i-1))) 1.05*max(X_U(t_zoom:tend_zoom,j + 3*(i-1)))])
% %     else
% %         axis([t_zoom*t_step tend_zoom*t_step 1.05*min(X_U(t_zoom:tend_zoom,j + 3*(i-1))) 1.05*max(X_U(t_zoom:tend_zoom,j + 3*(i-1)))])
% %     end
% %     grid on
% %     legend([legends(j + 3*(i-1)),legends(j+3)],'Interpreter','latex','FontSize',fontsize)
% %     ax = gca;
% %     ax.FontSize = fontsize;
% % end
% 
% %% STAMPA EQUAZIONI A SCHERMO
% close all
% 
% syms id iq wm vd vq Tl
% 
% % Definire i termini della libreria Theta
% Theta_sym = [id iq wm id*iq id*wm iq*wm vd vq Tl]; % id^2 iq^2 wm^2
% 
% % print equations
% eqn = -rs*wb/ld*id + wb*lq/ld*wm*iq + vd*wb/ld;
% eqn_start = vpa(eqn,2);
% 
% eqn = 0;
% i = 1;
% for j = 1:length(Theta_sym)
%     eqn = eqn + (Xi(j,i)) * Theta_sym(j);
% end
% eqn_SINDy = vpa(eqn,2);
% 
% disp('-');
% disp(['Eq1: ',char(eqn_start)]);
% disp(['Eq1: ',char(eqn_SINDy),' (SINDy)'])
% disp('-');
% 
% %
% eqn = -rs*wb/lq*iq -wb*ld/lq*wm*id - wb*phi_pu/lq*wm + vq*wb/lq;
% eqn_start = vpa(eqn,2);
% 
% eqn = 0;
% i = 2;
% for j = 1:length(Theta_sym)
%     eqn = eqn + (Xi(j,i)) * Theta_sym(j);
% end
% eqn_SINDy = vpa(eqn,2);
% disp(['Eq2: ',char(eqn_start)]);
% disp(['Eq2: ',char(eqn_SINDy),' (SINDy)']);
% disp('-');
% 
% %
% eqn = 3/2*pp*phi*Ib*iq/(2*Hm*Tb) - Tl/(2*Hm); %  + 3/2*pp*(Ld-Lq)/(2*Hm*Tb)*id*iq
% eqn_start = vpa(eqn,2);
% 
% eqn = 0;
% i = 3;
% for j = 1:length(Theta_sym)
%     eqn = eqn + (Xi(j,i)) * Theta_sym(j);
% end
% eqn_SINDy = vpa(eqn,2);
% 
% disp(['Eq3: ',char(eqn_start)]);
% disp(['Eq3: ',char(eqn_SINDy),' (SINDy)'])
% disp('-');
% 
% 
% %% Parameter estimation
% 
% syms wb_est ld_est lq_est rs_est psipm_est Hm_est
% 
% Xi_T = Xi';
% 
% ld_est = (double(wb/Xi_T(1,7)));
% lq_est = (double((wb/Xi_T(2,8) + Xi_T(1,6)/wb*ld_est)/2));
% rs_est = (double(-Xi_T(1,1)*ld_est/wb));
% Hm_est = abs(double(-1/(2*Xi_T(3,9))));
% psipm_est = (double(Xi_T(3,2)*2*Hm_est));
% 
% parameters_true = vpa([rs ld lq phi_pu Hm],2);
% parameters = vpa([rs_est ld_est lq_est psipm_est Hm_est],2);
% 
% comparison = [parameters; 
%                 parameters_true];
% vpa(comparison,2)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% START FROM HERE FOR: Feedforward NN 
% clc;
% clear;
% close all;
% 
% %% Motor parameters:
% pp = 5; % Number of pole pairs of the motor
% 
% Ld = 4.4*10^-3; % d-axis inductance in Henries (H)
% Lq = 4.4*10^-3; % q-axis inductance in Henries (H)
% Rs = 0.201;     % Stator resistance in Ohms (Ω)
% phi = sqrt(2)*0.0805; % Magnetic flux linkage (in Vs, Volts*seconds)
% 
% J = 9.8*10^-3;  % Rotor inertia (moment of inertia) in kg·m²
% Vn = 190;       % Nominal voltage in Volts (V)
% In = 10.2;      % Nominal current in Amps (A)
% wm_n = 2*pi*4500/60;  % Nominal mechanical speed in rad/s (converted from 4500 RPM)
% 
% % Current controller parameters for d-axis and q-axis with fc_I = 1000 Hz (Current
% % closed-loop cut-off frequency)
% KI_d = 29087;  % Integral gain for d-axis current controller
% Kp_d = 32.14;  % Proportional gain for d-axis current controller
% KI_q = 29087;  % Integral gain for q-axis current controller
% Kp_q = 32.14;  % Proportional gain for q-axis current controller
% 
% % % Uncomment this section if you are using fc_I = 500 Hz 
% % % Current controller parameters for d-axis and q-axis with fc_I = 500 Hz
% % KI_d = 7.5701e3;  % Integral gain for d-axis current controller
% % Kp_d = 16.0562;   % Proportional gain for d-axis current controller
% % KI_q = 7.5701e3;  % Integral gain for q-axis current controller
% % Kp_q = 16.0562;   % Proportional gain for q-axis current controller
% 
% KI_w = 13.15;     % Integral gain for speed controller
% Kp_w = 0.5777;    % Proportional gain for speed controller
% 
% % P.U. (Per Unit) base parameters
% Vb = sqrt(2)*Vn;   % Base voltage in the Per Unit system
% Ib = sqrt(2)*In;   % Base current in the Per Unit system
% wb = pp*wm_n;      % Base angular speed in rad/s (electrical speed)
% 
% Zb = Vb/Ib;        % Base impedance in Ohms (Ω)
% Lb = Zb/wb;        % Base inductance in Henries (H)
% phib = Vb/wb;      % Base magnetic flux in Vs (Volt-seconds)
% 
% rs = Rs/Zb;        % Per-unit stator resistance
% ld = Ld/Lb;        % Per-unit d-axis inductance
% lq = Lq/Lb;        % Per-unit q-axis inductance
% phi_pu = phi/phib; % Per-unit magnetic flux linkage
% 
% Tb = 3/2 * pp * phib * Ib;  % Base torque in Nm (Newton-meters)
% Hm = 0.5 * (J * wm_n^2) / (Tb * wm_n); % Normalized rotor inertia
% 
% % Reference values (to be adjusted for NN tuning)
% wm_slope = 0.025; % Reference mechanical speed slope (rad/s per time unit)
% id_ref = 3;       % Reference d-axis current
% Tl_cost = 1;      % Control load torque constant parameter
% k_friction = (0.05 * Tb) / (wm_n^2); % Friction torque, 5% of nominal torque at rated speed
% 
% % Noise parameters: set to zero for clean measurements
% sig_i = 0;  % (0.05/Ib)^2 Variance of current measurement noise (P.U.)
% sig_w = 0;   % (1/wm_n)^2 Variance of speed measurement noise (P.U.)
% 
% % Simulation step size and filter settings
% t_step = 1e-4;        % Sample time in seconds
% f_filter = 5000;      % Filter cutoff frequency in Hz
% 
% % Turn on/off filter flag (1 = on, 0 = off)
% filter_on = 0;
% 
% %% PARAMETRI SIMULAZIONE
% modello='PMSM_NN_def.slx';
% 
% % Set simulation
% Max_step_size=1e-3;
% Relative_tolerance=1e-5;
% 
% n_trajectories = 5;  % Number of trajectories (or simulations) to be generated
% input_data = [];     % Initialize input data array for the neural network
% output_data = [];    % Initialize output data array for the neural network
% 
% % % Reference values (parameters) for generating trajectories
% wm_slope_ref = linspace(0.01, 0.2, n_trajectories);  % Varying mechanical speed slopes
% id_ref_ref = linspace(0, 3, n_trajectories);         % Varying d-axis reference currents
% Cm_cost_ref = linspace(0.5, 4, n_trajectories);      % Varying control cost factors
% 
% stair_ramp_selector_ref = linspace(0, 1, n_trajectories);  % Varying selector between ramp and stair inputs
% stair_factor_ref = linspace(0, 1, n_trajectories);         % Varying factors for staircase input levels
% 
% StartTime = 0;    % Simulation start time in seconds
% FinishTime = 10;  % Simulation finish time in seconds
% 
% % Loop over each trajectory to simulate and collect data
% for j = 1:n_trajectories
% 
%     stair_factor = stair_factor_ref(j);                  % Set stair factor for this trajectory
%     stair_ramp_selector = stair_ramp_selector_ref(j);    % Set stair vs ramp selector for this trajectory
%     wm_slope = wm_slope_ref(j);                          % Set the mechanical speed slope for this trajectory
%     id_ref = id_ref_ref(j);                              % Set the d-axis reference current for this trajectory
%     Tl_cost = Cm_cost_ref(j);                            % Set the control cost for this trajectory
% 
%     % Run the simulation model 'modello' with parameters for this trajectory
%     simOut = sim(modello, 'StartTime', num2str(StartTime), 'Stoptime', num2str(FinishTime));
% 
%     % Retrieve simulation output data from the 'simout' variable (in per
%     % unit)
%     data = simOut.get('simout');
%     vd = data(:, 1);  % Voltage on d-axis
%     vq = data(:, 2);  % Voltage on q-axis
%     id = data(:, 3);  % Current on d-axis
%     iq = data(:, 4);  % Current on q-axis
%     wm = data(:, 5);  % Mechanical speed (omega)
%     Tl = data(:, 6);  % Load torque
% 
%     %  [id, iq, wm] - dq currents and speed
%     % [vd, vq, Cm] - dq voltages and torque command
%     y = [id iq wm];
%     u = [vd vq Tl];
% 
%     temp_input = [y , u];  % Concatenate current state and input into one array
% 
%     % Collect input/output data for neural network training
%     input_data = [input_data; temp_input(1:end-1, :)]; 
%     output_data = [output_data; y(2:end, :)];           
% 
%     clear temp_input  % Clear temporary variable to free memory
% 
%     j  % Display current iteration index (for tracking progress)
% end
% 
% % Convert the collected data into single-precision tensors for neural network input
% input_data = single(input_data);
% output_data = single(output_data);
% 
% %% Plot training set
% 
% legends = {'$i_{sd}$','$i_{sq}$','$\omega_{m}$','$v_{sd}$','$v_{sq}$','$T_{l}$'};
% fontsize=32;
% t_plot = t_step:t_step:FinishTime;
% sim_points = FinishTime/t_step;
% tiledlayout(3,2)
% 
% for j = 1:6
%     nexttile
%     for i = 1:n_trajectories
%         plot(t_plot',input_data( (i-1)*sim_points+1 : i*sim_points,j ),'LineStyle','-','LineWidth',1.5);
%         hold on
%     end
%     axis([StartTime FinishTime -1.05 1.05])
%     grid on
%     legend(legends(j),'Interpreter','latex','FontSize',fontsize,'Location','southeast')
%     xlabel('$[s]$','Interpreter','latex','FontSize',fontsize*2/3)
%     ax = gca;
%     ax.FontSize = fontsize*2/3;
% end
% 
% %% Get NN ready
% % Define a target MSE error threshold for stopping training
% target_mse = 1e-6;
% 
% % Update the training options to include the custom OutputFcn
% options = trainingOptions('adam', ...
%     'MaxEpochs', 8, ...
%     'MiniBatchSize', 32, ...
%     'InitialLearnRate', 0.001, ...
%     'Plots', 'training-progress', ...
%     'Verbose', false, ...
%     'OutputFcn', @(info) stopTrainingIfTargetMSEReached(info, target_mse));
% 
% layers = [
%     featureInputLayer(6, 'Normalization', 'none')       % 3 input neurons
%     fullyConnectedLayer(10)                               % 1° livello FC con 10 neuroni
%     sigmoidLayer                                         % Funzione di attivazione Sigmoid
%     fullyConnectedLayer(10)                               % 2° livello FC con 10 neuroni
%     reluLayer                                            % Funzione di attivazione ReLU
%     fullyConnectedLayer(10)                               % 3° livello FC con 10 neuroni
%     fullyConnectedLayer(3)                               % Livello finale con 3 output
%     regressionLayer];                                    % Livello di regressione per la perdita MSE
% 
% % Dividere i dati in input e output
% input_tensor = input_data;
% output_tensor = output_data;
% 
% %% Addestrare la rete
% net = trainNetwork(input_tensor, output_tensor, layers, options);
% 
% %% Test the NN
% 
% StartTime=0;
% FinishTime=10;
% 
% id_ref = 0; % MTPA operation
% Tl_cost = 6;
% 
% simOut = sim('PMSM_NN_Test_def.slx', 'StartTime', num2str(StartTime), 'StopTime', num2str(FinishTime));
% data = simOut.get('simout');
% 
% true_id = data(:,3);
% true_iq = data(:,4);
% true_wm = data(:,5);
% vd_NN=data(:,1);
% vq_NN=data(:,2);
% Cm_NN=data(:,6);
% 
% true_states = [true_id true_iq true_wm];  % True output to compare
% u_NN = [vd_NN, vq_NN, Cm_NN];
% 
% % input_comp = [true_states(1:end-1,:), u_NN(1:end-1,:)];
% 
% %%
% true_output = true_states(1:end-1,:);
% for i = 1:size(u_NN,1)
%     if i == 1
%         input_NN_states = true_states(1,:);
%     else
%         input_NN_states = predicted_output(i-1,:);
%     end
%     input_NN = [input_NN_states,u_NN(i,1:3)];
%     predicted_output(i,:) = predict(net,input_NN);
% end
% 
% %% Compare Test-set
% % Assuming predicted_output and true_output are of the same length
% time_vector = linspace(StartTime, FinishTime, size(predicted_output, 1));
% 
% % Plot id comparison
% figure;
% subplot(3,1,1)
% plot(time_vector', true_states(1:i,1), 'b', 'LineWidth', 2); hold on;
% plot(time_vector', predicted_output(:,1), 'r--', 'LineWidth', 2);
% xlabel('Time (s)');
% ylabel('i_d');
% title('Comparison of i_d (True vs Predicted)');
% legend('True', 'Predicted');
% grid on;
% 
% % Plot iq comparison
% subplot(3,1,2)
% plot(time_vector', true_states(1:i,2), 'b', 'LineWidth', 2); hold on;
% plot(time_vector', predicted_output(:,2), 'r--', 'LineWidth', 2);
% xlabel('Time (s)');
% ylabel('i_q');
% title('Comparison of i_q (True vs Predicted)');
% legend('True', 'Predicted');
% grid on;
% 
% % Plot wm comparison
% subplot(3,1,3)
% plot(time_vector', true_states(1:i,3), 'b', 'LineWidth', 2); hold on;
% plot(time_vector', predicted_output(:,3), 'r--', 'LineWidth', 2);
% xlabel('Time (s)');
% ylabel('\omega_m');
% title('Comparison of \omega_m (True vs Predicted)');
% legend('True', 'Predicted');
% grid on;