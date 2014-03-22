function [ output, out_net ] = feed_forward( nnet, input )
%FEED_FORWARD feeds-forward data through a 2 layer neural net
%   takes a nnet structure with theta1, theta2, and theta3
%   also has hid_units1, and hid_units_2
%   the theta# matrices should be of the form: to x from
    [~, num_examples] = size(input);
    layer0 = [input; ones(1, num_examples)]; % 1 is for bias
    current_layer = layer0;
    current_layer = sigmoid(nnet.theta{1} * current_layer);
    
    for i = 1:nnet.num_layers - 1
        nnet.hid_layers{i} = current_layer;
        current_layer = sigmoid(nnet.theta{i + 1} * [current_layer; ones(1, num_examples)]);
    end
    
    output = current_layer;
    out_net = nnet;
end

