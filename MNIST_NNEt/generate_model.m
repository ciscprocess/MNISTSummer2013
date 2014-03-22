function [ model ] = generate_model(n_layers, n_inputs, hids, n_out, damp )
%GENERATE_MODEL generates a new 2-layer nnet
%   Detailed explanation goes here
    nnet = {};
    
    nnet.num_inputs = n_inputs;
    nnet.num_layers = n_layers;
    nnet.hidden_sizes = hids;
    nnet.num_outputs = n_out;
    
    nnet.hid_layers = {};
    nnet.theta = {};
    hids = [hids n_out];
    nnet.theta{1} = (rand(hids(1), n_inputs + 1) - 0.5) / damp;
    
    for i = 1:n_layers - 1
        nnet.hid_layers{i} = zeros(hids(i), 1);
        nnet.theta{i + 1} = (rand(hids(i + 1), hids(i) + 1) - 0.5) / damp;
    end
    
    model = nnet;
end