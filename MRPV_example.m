
load('GSE52583.mat')
kfold = 10;
Indices = crossvalind('Kfold', class, kfold);
predicted_class = zeros(size(class));
RPdim = 50;
RPspaces = 10;
Fitopt = 'RLDA';

for i_fol = 1:kfold
    data_train = data(Indices~=i_fol,:);
    data_test  = data(Indices==i_fol,:);
    labels = class(Indices~=i_fol);
    predicted_class(Indices==i_fol) = MRPV(data_test, data_train, labels, RPdim, RPspaces, Fitopt);
end

% sum(class==predicted_class)/length(class)