function [ data, targets ] = readAna()
%READANA Summary of this function goes here
%   Detailed explanation goes here
    fid = fopen('pima_diabetes.dat');
    tline = fgetl(fid);
    targets = [];
    data = [];
    while ischar(tline)
        tline = fgetl(fid);
        if tline == -1
            break
        end
        t = strsplit(tline, ',');
        targets = [targets str2double(t(1))];
        d = [];
        for i = t(2:end)
            d = [d str2double(i)];
        end
        
        data = [data d'];
    end

    fclose(fid);

end

