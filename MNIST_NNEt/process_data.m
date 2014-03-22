function [ flattened, binaried ] = process_data( imgs, labs )
%PROCESS_DATA Summary of this function goes here
%   Detailed explanation goes here
    [~, ~, no_imgs] = size(imgs);
    flattened = zeros(400, no_imgs);
    binaried = zeros(10, no_imgs);
    for i = 1:no_imgs
        temp = imgs(:, :, i);
        flattened(:, i) = temp(:);
        vec = zeros(10, 1);
        tmp = de2bi(2^labs(i));
        [~, n] = size(tmp);
        vec(1:n, 1) = tmp;
        binaried(:, i) = vec;
    end
end

