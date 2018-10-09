function PCA_ksvd_max( inpath, outpath, meanX, coef, whiten, U, G)
nonzero = 32;

A = dlmread(inpath);

% l2 norm
for i = 1 : size(A, 1)
	A(i,:) = A(i,:) / norm(A(i,:));
end

% PCA
A = bsxfun(@minus, A, meanX);
SCORE = A * coef;
SCORE = bsxfun(@rdivide, SCORE, whiten);

% l2 norm
for i = 1 : size(SCORE, 1)
	SCORE(i,:) = SCORE(i,:) / norm(SCORE(i,:));
end

% sparse
A = SCORE';
V = omp(U'*A, G, nonzero);
V = full(V);
V = V';
for i = 1:size(V, 1)
    V(i, :) = V(i, :) / norm(V(i, :));
end
save(outpath, 'V');
end


