despath = dir('gongsi1000SURFtxt\*.txt');
len = size(despath, 1);
len
for i = 1:len
    des = dlmread(['gongsi1000SURFtxt' '\\' despath(i).name]);
    des = (des==0);
    for j = 1:size(des,1)
        tmp = des(j, :);
        if size(tmp(tmp==1), 2) == 64
            despath(i).name
        end
    end
end

% data = dlmread('sample_with_kp.txt');
% z = (data==0);
% for i = 1:size(z, 1)
%     tmp = z(i, :);
%     if size(tmp(tmp == 1), 2) == 64
%         i
%     end
% end