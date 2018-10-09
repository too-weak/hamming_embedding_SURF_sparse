function AssignMemberShip(outcenters,data,sparsity,outFolder)

sparsity = str2num(sparsity);
assginmedFolder = [outFolder,'\assginMed.txt'];
disFolder = [outFolder,'\dis.txt'];

D = dlmread(outcenters);
X = dlmread(data);
D = D';
X = X';

gamma = omp(D'*X, D'*D, sparsity);

[i,j,s]=find(gamma);

tmpi=reshape(i,sparsity,size(i,1)/sparsity);
tmpi=tmpi';

tmps=reshape(s,sparsity,size(s,1)/sparsity);
tmps=abs(tmps);
maxtmp = max(max(tmps));
tmps = maxtmp-tmps;
tmps=tmps';

[tmps ind]=sort(tmps,2);
for ii=1:size(tmps,1)
    tmpi(ii,:)=tmpi(ii,ind(ii,:));
end;

dlmwrite(assginmedFolder,tmpi,'delimiter',' ','newline','pc');
dlmwrite(disFolder,tmps,'delimiter',' ','newline','pc');

end

