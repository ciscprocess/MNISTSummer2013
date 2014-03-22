function [ new_net ] = sample_errors( nnet, inputs, targets, samples )
%SAMPLE_ERRORS Summary of this function goes here
%   Detailed explanation goes here
    minerr = Inf;
    minmodel = 0;
    
    for i = 1:samples
        outputs = feed_forward(nnet, inputs);
        terr = calc_model_error(outputs, targets);
        fprintf('Temp error: %f\n', terr);
        if terr < minerr
            minerr = terr;
            minmodel = nnet;
        end
        nnet = generate_model(400, 100, 20, 10, 2);
    end
    
    new_net = minmodel;
end

