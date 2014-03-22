function [ errs, percent ] = evaluate_model( nnet, inputs, targets )
%EVALUATE_MODEL Summary of this function goes here
%   Detailed explanation goes here
    outputs = feed_forward(nnet, inputs);
    [~, no_outs] = size(outputs);
    missed = 0;
    for i = 1:no_outs
        %[~, img_class] = max(outputs(1:10, i));%round(log(bi2de(round(outputs(1:10, i))'))/log(2));
        img_class = round(outputs(i));
        %img_class = img_class - 1;
        if img_class ~= targets(i)
            missed = missed + 1;
        end
    end
    errs = missed;
    percent = missed / no_outs;
    fprintf('Percentage error: %f', percent);
end

