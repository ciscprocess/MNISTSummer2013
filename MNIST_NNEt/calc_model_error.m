function [ err ] = calc_model_error( outputs, targets )
%CALC_MODEL_ERROR Summary of this function goes here
%   Detailed explanation goes here
    err = 0.5 * sum(sum((targets - outputs) .^ 2));
end