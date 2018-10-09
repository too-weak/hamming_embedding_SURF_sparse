function ksvd_train(input, output)
    dim = 4;
    X=importdata(input);
    disp('finish reading data,start computing...');

    %l2 norm
    for i = 1 : size(X, 1)
        X(i,:) = X(i,:) / norm(X(i,:));
    end

    %PCA train
    [coef,S,latent] = pca(X);

    cum = cumsum(latent);
    S = S(:,1:dim);
    coef = coef(:,1:dim);

    Cov = nancov(S);
    d = sqrt(diag(Cov));
    whiten = d';

    S = bsxfun(@rdivide, S, whiten);
    meanX = mean(X, 1);

    %l2 norm
    for i = 1 : size(S, 1)
        S(i,:) = S(i,:) / norm(S(i,:));
    end

    % KSVD train
    addpath ksvd
    addpath omp
    X = S';
    params.data = X;
    params.Tdata = 32;
    params.dictsize = 1024;
    params.iternum = 50;
    params.memusage = 'high';


    [U,g,err] = ksvd(params,'i');

    save(output, 'U', 'whiten', 'coef', 'meanX');
end

