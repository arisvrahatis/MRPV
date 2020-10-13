function prediction = MRPV(data_test, data_train, labels, RPdim, RPspaces, Fitopt)
%function prediction = rpknnvote(data_test, data_train, labels, rp_dim, rp_spaces, k)
%   Predicts labels for data_test using random subspaces and Schulze voting.
%   
% Input
% data_test  : Test Data (Matrix: Samples x Features)
% data_train : Train Data (Matrix: Samples x Features)
% labels     : Train Labels (Vector: Samples x 1)
% RPdim      : Dimension of the projected space (RPdim << Features) (default: RPdim = 50)
% RPspaces   : Number of projected spaces (default: RPspaces = 10)
% Fitopt     : Classifier selection (Options: 'KNN','LDA','RLDA') (default: 'RLDA')
%    
% Output:
% prediction : Test Labels (Vector: Samples x 1)

%   Copyright 2020: A.G. Vrahatis, S. Tasoulis

    if nargin == 3
        RPdim = 50; RPspaces = 10; Fitopt = 'RLDA';
    elseif nargin == 4
        RPspaces = 10; Fitopt = 'RLDA';
    elseif nargin == 5
        Fitopt = 'RLDA';
    end

    Nsamples = size(data_test, 1);
    Nfeatures = size(data_train, 2);
    
    pred_all = zeros(Nsamples, RPspaces);
    % Random Projected Spaces
    R = randn(Nfeatures, RPdim * RPspaces);
    % Projection (one mulitplication for all spaces)
    B_tr = data_train * R;
    B_ts = data_test  * R;
    
    %for each subspace, find k nearest neighbours and compute class percentages
    for rp = 1:RPspaces
        % Projection (we create all multiplications in one matrix)
        d_tr  = B_tr(:, (rp * RPdim - RPdim + 1):(rp * RPdim));
        d_ts  = B_ts(:, (rp * RPdim - RPdim + 1):(rp * RPdim));
            
        switch Fitopt
            case 'KNN'
                fitmodel = fitcknn(d_tr, labels, 'NumNeighbors', 5, 'Distance', 'cityblock');
            case 'LDA'
                fitmodel = fitcdiscr(d_tr, labels, 'DiscrimType', 'diaglinear');
            case 'RLDA'
                fitmodel = fitcdiscr(d_tr, labels);
        end              
        pred_all(:, rp) = predict(fitmodel, d_ts); 
    end
    prediction = mode(pred_all');
end

