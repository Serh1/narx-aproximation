%% SI - Part 2
clear all
close all
% Load the given dataset
load('iddata-20.mat')

figure
plot(id)
figure
plot(val)

% Take out the Input-Output for each Identification and Validation data
u_id = id.InputData;
y_id = id.OutputData;

u_val = val.InputData;
y_val = val.OutputData;

N = length(u_val);

% We built a function that uses the given dataset and the parameters na nb
% an m. For simplify we took nk = 0 and na = nb.

% The function is called for each m and na from 1 to 4.
% At each set of parameters an MSE will be calculated and saved in a table.

for m=1:4
    for na=1:4
        nb = na;
        [y_iden,y_prediction,y_simulation]=nonlinearARX(m,na,nb,u_id,y_id,u_val,y_val);
        MSE_pred(m,na)=(1/N)*sum((y_prediction-y_val).^2);
        MSE_sim(m,na)=(1/N)*sum((y_simulation-y_val).^2);
    end
end

% Find the parameters m and na at which the MSE is minimum from the table.
[Mp Nap] = find(min(min(MSE_pred)) == MSE_pred);
[Ms Nas] = find(min(min(MSE_sim)) == MSE_sim);

Nbp=Nap;
% Recall the function with the best fitted parameters 
% to plot our final solution.
[y_iden,y_p,y_s]=nonlinearARX(Mp,Nap,Nbp,u_id,y_id,u_val,y_val);

figure;
aprox_identification=iddata(y_iden,u_id,id.Ts);
compare(id,aprox_identification);
title(['Best prediction on identification for m=',num2str(Mp),'  na=nb=',num2str(Nap),'  nk=0']);

figure;
aprox_validation=iddata(y_p,u_val,val.Ts);
compare(val,aprox_validation);
title(['Best prediction on validation for m=',num2str(Mp),'  na=nb=',num2str(Nap),'  nk=0']);

Nb=Nas;
% Because in our case the best na and m are the same for Prediction and
% Simulation we don't plot another figure for best simulation on
% identification.
[y_iden,y_p,y_s]=nonlinearARX(Ms,Nas,Nb,u_id,y_id,u_val,y_val);

figure;
aprox_simulation=iddata(y_s,u_val,val.Ts);
compare(val,aprox_simulation);
title(['Best simulation for m=',num2str(Ms),', na=nb=',num2str(Nas),'  nk=0']);

% Plotting the Real vs Prediction vs Simulation aproximations
figure;
hold;
plot(y_val);
plot(y_p);
plot(y_s);
legend("Validation","Prediction","Simulation");
title(['Best fit for m=',num2str(Ms),', na=nb=',num2str(Nas),'  nk=0']);

function [Y_Identification,Y_Prediction,Y_Simulation]= nonlinearARX(m,na,nb,u_id,y_id,u_val,y_val)
    %% Prediction
    % Construct the Identification matrix y(k-na) u(k-nb) 
    l_id = length(u_id);
    l_val = length(u_val);
    n = na*2;
    temp1 = [];
    temp2 = [];
    yid = zeros(length(y_id),na);
    uid = zeros(length(u_id),nb);
    for k = 1:length(y_id)
        for i = 1:na
            if(k-i)<=0
                yid(k,i) = 0;
                uid(k,i) = 0;
            else
                yid(k,i) = -y_id(k-i);
                uid(k,i) = u_id(k-i);
            end
        end
    end
    did = [yid uid];
    % Construct the powers matrix 
    P = cell(1, n);
    % Taking all the combinations possible for the given m
    [P{:}] = ndgrid(0:m);
    P = reshape(cat(n+1, P{:}), [], n);
    P=fliplr(P);
    % Verify and eliminate the elements that 
    % doesn't agree with the condition
    for i=1:length(P)
        if sum(P(i,:))>m
            P(i,:)=0;
        end
    end
    
    % It also deletes the all zero row so we'll merge it back
    P(all(~P,2),:)=[];
    P=[zeros(1,n); P];
    l_P=length(P);

    % Rise the each row to each power row and save it into a cell vector
    cells1 = cell(l_P,1);
    for j=1:l_P
        for i = 1:l_id
            temp1(i,:) = did(i,:).^P(j,:);
        end
        cells1{j} = temp1;
    end

    PHI_id=zeros(l_P,l_id);
    % PHI = [phi1 phi2 phi3 ...   ]
    % Product by colomn to get the PHI regressor
    for i = 1:l_P
        for j=1:l_id
            PHI_id(i,j) = prod(cells1{i}(j,:));
        end
    end
    THETA = PHI_id'\y_id;
    % Identification  
    Y_Identification=PHI_id'*THETA;

    % Construct the Validation matrix y(k-na) u(k-nb) 
    yval = zeros(l_val,na);
    uval = zeros(l_val,nb);
    for k = 1:l_val
        for i = 1:na
            if(k-i)<=0
                yval(k,i) = 0;
                uval(k,i) = 0;
            else
                yval(k,i) = -y_val(k-i);
                uval(k,i) = u_val(k-i);
            end
        end     
    end
    dval = [yval uval];
   
    % Repeat the previous steps for validation this time
    cells2 = cell(l_P,1);
    for j=1:l_P
        for i = 1:l_val
            temp2(i,:) = dval(i,:).^P(j,:);
        end
        cells2{j} = temp2;
    end

    PHI_val=zeros(l_P,l_val);
    for i = 1:l_P
        for j=1:l_val
            PHI_val(i,j) = prod(cells2{i}(j,:));
        end
    end
    % Obtain the prediction
    Y_Prediction = PHI_val' * THETA;

    %% Simulation
    Y_Simulation=zeros(1,l_val);
    X=[];XMat=[];PHI=[];
    % Calculate Simulation at each step by using 
    % the previously calculated value 
    for k=1:l_val
        for i=1:na
            if (k-i)<=0
                X(i)=0;
                X(i+na)=0;
            else
                X(i)=-Y_Simulation(k-i);
                X(i+na)=u_val(k-i);
            end
        end
        XMat(k,:)=X;
        for j=1:l_P
            PHI(k,j)=prod(XMat(k,:).^P(j,:))*THETA(j,1);
        end
        Y_Simulation(k)=sum(PHI(k,:));
    end
    Y_Simulation=Y_Simulation';
end




