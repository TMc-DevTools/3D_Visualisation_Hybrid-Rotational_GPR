% Numerical Implementation of Combined Processing Methodology (CPM)

% Function to Compute and Plot CPM Response 
function [t,f,X,Y,Z] = gpr360_CPM(filename)
    % -------------------------------------------------
    % Returns CPM temporal response profile for a 1D trace. 

    % Parameters: 
    % filename (string): input trace file 

    Returns: 
    % t (array): Temporal domain values [s]
    % f (array): frequency domain values [Hz]
    % X (array): Normalised windowed IFFT temporal response profile  
    % Y (array): Normalised MUSIC temporal response profile
    % Z (array): Normalised CPM temporal response profile

    % -------------------------------------------------

%% Configuration 

% Set Default Input File 
if (nargin < 1)
   filename = 'VNAtr08.txt';
elseif strcmp(filename,'VNAtr00.txt')
    error(
        'Error: You cannot apply calibrated CPM to this trace')
end 

% Parameters 
window = 'KB';      % 'Uniform','Nuttall','KB','Ham','Hann'
type = 'MSSP';      % 'Raw','SSP','MSSP' (case Sensitive)
p = 0.1;            %  Threshold for noise subspace estimation 
T_plot = 40e-9;     %  Maximal response time to show on plots
M = 10;             %  Number of snapshots SSP/MSSP

% Import 1D Trace
[t,f,X] = gpr360_ifft(filename,window);
[~,~,Y] = gpr360_MUSIC(filename,type,p,M);

% Linked Parameters                 
dt = t(2)-t(1); 
J = size(X,1);

% Apply Boundary Conditions For 4th Order Central-Difference Method
% t<=0   | Fixed     | X=0,Y=0        
% Signal Takes Time to Return (i.e Initially No Return Amplitude)
% t_End  | Periodic  | X(End+1)=X(1) 
% Listing Interval Is Finite Sample of Potentially Infinite Duration 
X_bc = [0;0;X;X(1);X(2)];      
Y_bc = [0;0;Y;Y(1);Y(2)]; 

% CPM Storage Array
Z = zeros(J,1); 

%% Apply CPM
for j = ((1:J)+2)
    dXdt = (X_bc(j-2)-8*X_bc(j-1)+8*X_bc(j+1)-X_bc(j+2))/(12*dt); 
    dYdt = (Y_bc(j-2)-8*Y_bc(j-1)+8*Y_bc(j+1)-Y_bc(j+2))/(12*dt);
    Z1 = (X_bc(j)*abs(dYdt)) + (Y_bc(j)*abs(dXdt));
    Z2 = sqrt(abs(dXdt)^2 + abs(dYdt)^2);
    Z(j-2,1) = Z1./Z2;
end 

% Renormalise
Z = normalize(abs(Z),'range');

%% Plot Temporal Response 

% Configure Plotter
set(0,'defaultTextInterpreter','latex');
figure();  

% Uncomment As Required

% % Plot IFFT Only 
% subplot(2,3,1)
% grid on; hold on; 
% grid(gca,'minor');
% plot(t,X,'-.k'); 
% legend('$x_{8}(t)|$(IFFT-Kaiser)','Interpreter','latex'); 
% xlabel('Time (ns)'); 
% ylabel('Amplitude (Normalised)');  
% xlim([0,T_plot]);
% xticks([0,10e-9,20e-9,30e-9,40e-9])
% xticklabels({'0','10','20','30','40'})
% 
% % Plot MUSIC Only 
% subplot(2,3,4)
% grid on; hold on; 
% grid(gca,'minor');
% plot(t,Y,'-r'); 
% legend('$y_{8}(t)|$(MUSIC-MSSP)','Interpreter','latex'); 
% xlabel('Time (ns)'); 
% ylabel('Amplitude (Normalised)'); 
% xlim([0,T_plot]);
% xticks([0,10e-9,20e-9,30e-9,40e-9])
% xticklabels({'0','10','20','30','40'})
% 
% % Plot MUSIC And CPM
% subplot(2,3,2)
% grid on; hold on; 
% grid(gca,'minor');
% plot(t,Z,'-bo','MarkerSize',3);
% plot(t,Y,'-r'); 
% legend('$z_{8}(t)|$(CPM)','$y_{8}(t)|$(MUSIC-MSSP)',
            'Interpreter','latex'); 
% xlabel('Time (ns)'); 
% ylabel('Amplitude (Normalised)'); 
% xlim([0,T_plot]);
% xticks([0,10e-9,20e-9,30e-9,40e-9])
% xticklabels({'0','10','20','30','40'})
% 
% % Plot IFFT And CPM
% subplot(2,3,5)
grid on; hold on; 
grid(gca,'minor');
plot(t,Z,'-bo','MarkerSize',3);
plot(t,X,'-.k'); 
legend('$z_{8}(t)|$(CPM)','$x_{8}(t)|$(IFFT-Kaiser)',
            'Interpreter','latex'); 
xlabel('Time (ns)'); 
ylabel('Amplitude (Normalised)'); 
xlim([0,T_plot]);
xticks([0,10e-9,20e-9,30e-9,40e-9])
xticklabels({'0','10','20','30','40'})

% Plot CPM Only
% subplot(2,3,[3,6])
% grid on; hold on; 
% grid(gca,'minor');
% plot(t,Z,'-bo','MarkerSize',4);
% legend('$z_{8}(t)|$(CPM)','Interpreter','latex'); 
% xlabel('Time (ns)'); 
% ylabel('Amplitude (Normalised)'); 
% xlim([0,T_plot]);
% xticks([0,10e-9,20e-9,30e-9,40e-9])
% xticklabels({'0','10','20','30','40'})

end

% Function to Compute Windowed Fast Fourier Transform (IFFT)
function [t,f,X] = gpr360_ifft(xf1_filename,window)
    % -------------------------------------------------
    % Returns IFFT temporal response profile for a 1D trace. 

    % Parameters: 
    % xf1_filename (string): input trace file
    % window (string): Type of window function 
                           % 'None', 'Ham', 'Hann', 'KB', 'Nuttall'

    Returns: 
    % t (array): Temporal domain values [s]
    % f (array): frequency domain values [Hz]
    % X (array): Normalised windowed IFFT temporal response profile  

    % -------------------------------------------------
    
%% Configuration 

% Parameters 
parameters = importdata('Parameters.txt'); 
f_min = parameters.data(1)*(1e9); 
f_max = parameters.data(2)*(1e9);  

% Import Background Calibration Data
xf0 = importdata(xf0_filename); 
xf0 = xf0(:,1) + (1i)*xf0(:,2);

% Import Main Data
xf1 = importdata(xf1_filename); 
xf1 = xf1(:,1) + (1i)*xf1(:,2);

% Linked Parameters
L = size(xf1,1);
df = (f_max-f_min)/(L-1);
T = 1/df; 
dt = T/(L-1); 

% Get Temporal & Frequency Values 
t = (0:dt:(L-1)*dt)';
f = (f_min:df:f_max)';

%% Apply Windowing Function BEFORE Transformation

% Define Window Function
if strcmp(window,'Ham')                                                     
    a0 = 25/46;                                                             
    a1 = 1-a0;
    w = a0 - a1.*cos((2.*pi.*(0:L-1)')./(L-1));  
elseif strcmp(window,'Hann')                                                
    a0 = 0.5;                                                               
    a1 = 1-a0;
    w = a0 - a1.*cos((2.*pi.*(0:L-1)')./(L-1)); 
elseif strcmp(window,'KB')                                                  
    a0 = 3; 
    w_top = besselj(0,pi*a0*sqrt(1-((2*(0:L-1)')/(L-1)-1).^2));
    w_bot = besselj(0,repelem(pi.*a0,L)'); 
    w = w_top./w_bot;
elseif strcmp(window,'Nuttall')                                             
    a0 = 0.355768;
    a1 = 0.487396;
    a2 = 0.144232; 
    a3 = 0.012604;
    w = a0 - a1.*cos((2.*pi.*(0:L-1)')./(L-1)) + ...
        a2.*cos((4.*pi.*(0:L-1)')./(L-1)) - ...
        a3.*cos((6.*pi.*(0:L-1)')./(L-1)); 
else                                                                        
    w = repelem(1,L)';
end 

% Convolve Window                
xf0 = conv(xf0,w,'same');                                                     
xf1 = conv(xf1,w,'same');           

%% Perform IFFT

% Inverse Fast Fourier Transform 
xt0 = ifft(xf0);
xt1 = ifft(xf1); 

% Background Calibration Subtraction 
xt = xt1 - xt0; 

% Normalised Output  
X = normalize(abs(xt),'range'); 

end 

% Function to Compute Multiple Signal Classification (MUSIC)
function [t,f,Y] = gpr360_MUSIC(xf1_filename,type,p,M) 
    % -------------------------------------------------
    % Returns MUSIC temporal response profile for a 1D trace. 

    % Parameters: 
    % xf1_filename (string): input trace file
    % type (string): Type of spatial smoothing function 
                           % 'None', 'SSP', 'MSSP'
    % p (float): Noise subspace threshold tolerance 
                    - within 100*p% of max eigenvalue
    % M (integer): Number of snapshots to use in SSP/MSSP

    Returns: 
    % t (array): Temporal domain values [s]
    % f (array): frequency domain values [Hz] 
    % Y (array): Normalised MUSIC temporal response profile

    % -------------------------------------------------

%% Configuration 

% Parameters 
parameters = importdata('Parameters.txt'); 
f_min = parameters.data(1)*(1e9); 
f_max = parameters.data(2)*(1e9);  

% Import Background Calibration Data
xf0 = importdata(xf0_filename); 
xf0 = xf0(:,1) + (1i)*xf0(:,2);

% Import Main Data
xf1 = importdata(xf1_filename); 
xf1 = xf1(:,1) + (1i)*xf1(:,2);

% Linked Parameters
L = size(xf1,1);
df = (f_max-f_min)/(L-1);
T = 1/df; 
dt = T/(L-1); 

% Get Frequency & Temporal Values
f = (f_min:df:f_max)';
t = (0:dt:(L-1)*dt)';

% Number of Scatting Signals/Timesteps 
K = L; 

%% Step 1: Approximate Signal Covariance Matrix

if strcmp(type,'None')                                                      
    S0 = xf0*xf0';                                                                
    S1 = xf1*xf1'; 
elseif strcmp(type,'SSP')                                                  
    N = L-M+1;
    S0 = zeros(N);
    S1 = zeros(N);
    for k = 1:M
        xk0 = xf0(k:N+(k-1),1);
        xk1 = xf1(k:N+(k-1),1); 
        Sk0 = xk0*xk0'; 
        Sk1 = xk1*xk1'; 
        S0 = S0 + Sk0;
        S1 = S1 + Sk1;
    end 
    S0= S0./M; 
    S1= S1./M;
elseif strcmp(type,'MSSP')                                              
    N = L-M+1;
    J = fliplr(eye(N));                                                         
    S0 = zeros(N);
    S1 = zeros(N); 
    for k = 1:M
        xk0 = xf0(k:N+(k-1),1); 
        xk1 = xf1(k:N+(k-1),1);
        Sk0 = J*(xk0*xk0')'*J;
        Sk1 = J*(xk1*xk1')'*J;
        S0 = S0 + Sk0 + (J*Sk0*J); 
        S1 = S1 + Sk1 + (J*Sk1*J); 
    end 
    S0 = S0./(2*M); 
    S1 = S1./(2*M);
end 
    
%% Step 2: Eigenvalue Decompose (EVD)

[V0,lambda0] = eig(S0,'vector');                                                   
[lambda0,ind0] = sort(lambda0,'ComparisonMethod','abs');                        
V0 = V0(:,ind0);

[V1,lambda1] = eig(S1,'vector');                                                   
[lambda1,ind1] = sort(lambda1,'ComparisonMethod','abs');                        
V1 = V1(:,ind1);

%% Step 3: Noise Subspace Estimation 

En0 = V0(:,abs(lambda0)<(1-p)*max(lambda0)); 
En1 = V1(:,abs(lambda1)<(1-p)*max(lambda1));

%% Step 4: Compute Pseudospectrum 

if strcmp(type,'Raw')

    % Define Steering Matrix
    A = exp(-(2*pi).*(1i).*f*t'); 

    % Create Storage Array 
    xt0 = zeros(K,1);
    xt1 = zeros(K,1);

    % Iterate Computation of Pseudospectrum 
    for k = 1:K 
        a = A(:,k); 
        xt0(k,1) = (a'*a)/(a'*(En0*En0')*a);  
        xt1(k,1) = (a'*a)/(a'*(En1*En1')*a); 
    end 

elseif or(strcmp(type,'SSP'),strcmp(type,'MSSP'))
    
    % Define Updated Steering Matrix
    A = exp(-(2*pi).*(1i).*f*t');                                               
    D = diag(exp(-(2*pi).*(1i).*df.*t'));                                   
    B = A*(D^(K-1));
    
    % Create Storage Array
    xt0 = zeros(K,1); 
    xt1 = zeros(K,1);
    
    % Iterate Computation of Pseudospectrum (SSP/MSSP)
    for k = 1:K    
        b = B(1:N,k); 
        xt0(k,1) = (b'*b)/(b'*(En0*En0')*b);
        xt1(k,1) = (b'*b)/(b'*(En1*En1')*b);
    end
    
end 

% Background Calibration Subtraction 
xt = xt1 - xt0;  

% Normalised Output  
Y = normalize(abs(xt),'range');

end 


