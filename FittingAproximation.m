%% Fitting an Unknown Function - SI Project
% Members: Mateiu Sergiu, Popescu Teodora, Talpos Diana
% Group: 30331

clear all
% Load the Model data index = 20
load('proj_fit_20.mat');

% Save the Input and Output data for each Identification 
% and Validation data set:
X_id(1,:) = id.X{1,1};
X_id(2,:) = id.X{2,1};
Y_id = id.Y;
l_id = length(X_id);

X_val(1,:) = val.X{1,1};
X_val(2,:) = val.X{2,1};
Y_val = val.Y;
l_val = length(X_val);

% Plot the identification data 
mesh(X_id(1,:),X_id(2,:),Y_id),title("Identification Data Set");

% Create initial variables with zero values
% We saved every THETA and PHI_val matrixes into cell variables. Such that
% at the final plot we can just take the proper value instead of
% recalculating the solution for the specific m
MAX = 30;
MSE_Array = zeros(2,MAX);
THETA = cell(1,MAX);
PHI_Cell = cell(1,MAX);

for m = 1:MAX
PHI_id=zeros(l_id,m);
PHI_val=zeros(l_id,m);
% Construction of the Phi powers a and b
a = 0:m;
b = 0:m;
[p,c] = ndgrid(a,b); 
AB = [p(:),c(:)];

% Remove the values that don't respect the condtion
for k = 1:length(AB)
    if AB(k,1) + AB(k,2) > m
        AB(k,1) = 0;
        AB(k,2) = 0;
    end
end
AB( all(~AB,2), : ) = [];
AB = [0,0 ; AB];

% Compute Phi matrix with the identification data
c = 0;
 for i=1:l_id 
     for j=1:l_id 
         c = c+1;
        for k=1:length(AB)
            PHI_id(c,k) =(X_id(1,i)^AB(k,1))*(X_id(2,j)^AB(k,2));         
        end
     end   
 end
% Transform the given output matrix in the colomn matrix so we can
% calculate further the THETA matrix
Y_id = Y_id(:);
THETA{1,m} = PHI_id\Y_id;
Y_hat_id = PHI_id*THETA{1,m};

%Transform the aproximated and the given outputs back into matrix form 
Y_hat_id = reshape(Y_hat_id,l_id,l_id);
Y_id = reshape(Y_id,l_id,l_id);

% Calculate MSE for identification data
Error_id = abs(Y_hat_id-Y_id).^2;
MSE_val = sum(Error_id(:))/numel(Y_hat_id);
MSE_Array(1,m) = MSE_val;

% Compute Phi matrix with the validation data
c = 0;
 for i=1:l_val 
     for j=1:l_val
         c = c+1;
        for k=1:length(AB)
            PHI_val(c,k) = (X_val(1,i)^AB(k,1))*(X_val(2,j)^AB(k,2));         
        end
     end   
 end
 
% Save every PHI value into a Cell variable depending on m
PHI_Cell{1,m} = PHI_val; 

%Transform the aproximated and the given outputs back into matrix form 
Y_hat = PHI_val*THETA{1,m};
Y_hat = reshape(Y_hat,l_val,l_val); 

% Calculate MSE for validation data
Error_val = abs(Y_hat-Y_val).^2;
MSE_val = sum(Error_val(:))/numel(Y_hat);
MSE_Array(2,m) = MSE_val;
end

figure
% Calcualte the minimum MSEs from the Array of errors
MIN_MSE = [min(MSE_Array(1,:)) min(MSE_Array(2,:))];

% Find the index at each minimum error
Min_M_val1 = find(MSE_Array(1,:) == MIN_MSE(1));
Min_M_val2 = find(MSE_Array(2,:) == MIN_MSE(2));
Min_M_val = [Min_M_val1 Min_M_val2];
m = 1:MAX;

% Plot the minimum MSEs in respect to degree m
plot(m,MSE_Array(1,:),m,MSE_Array(2,:))
hold
plot(Min_M_val(1),MIN_MSE(1),'*',Min_M_val(2),MIN_MSE(2),'*')
title("Identification MSEs vs Validation MSEs")
xlabel('Degree m');ylabel('Error Value')
legend('Identification MSE','Validation MSE','Minimum Identification MSE','Minimum Validation MSE')

% Calculate the best aproximation for the minimum error 
Y_hat_min = PHI_Cell{1,Min_M_val(2)}*THETA{1,Min_M_val(2)};
Y_hat_min = reshape(Y_hat_min,l_val,l_val);

% Plot the most optimal solution
figure
mesh(X_val(1,:),X_val(2,:),Y_hat_min,'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','r')
hold on
mesh(X_val(1,:),X_val(2,:),Y_val,'FaceColor','none', 'FaceAlpha',0.5, 'EdgeColor','k')
title("Approximated Model for m="+Min_M_val(2))
legend("Approximated","Validation")
 
