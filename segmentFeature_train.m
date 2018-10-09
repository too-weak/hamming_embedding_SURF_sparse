function segmentFeature_train()
    % sparse frame feature
    addpath omp;
    addpath ksvd;
    configFile = 'config\config.txt';          % config file
    mainDir = 'resource\gongsi1000SURFtxt\';         % input feature
    outDir = 'resource\segment\';              % out feature
    inputMat = 'data\\trainmodel.mat';                       % train mat
    [name, format] = textread(configFile, '%s %s');

    mkdir(outDir);
    load(inputMat);

    subdir  = dir( mainDir );
    count = 0;
    namelist = cell(0);
    total = [0];
    G = U' * U;
    for i = 1 : length( subdir )
        if(subdir( i ).isdir)               
            continue;
        end

        datpath = fullfile( mainDir, subdir( i ).name);
        outdatpath = fullfile(outDir, subdir( i ).name(1:end - 4));
        outdatpath = [outdatpath '.mat'];

        count = count + 1;
        PCA_ksvd_max(datpath, outdatpath, meanX, coef, whiten, U, G);
    end
end
