function [ gradients ] = calc_gradient( nnet, input, target )
% calc_gradient Calculates the gradient of the weights for a given neural
% network model

    [output, nnet] = feed_forward(nnet, input);
    [~, num_examples] = size(input);
    dE_dYOut = output - target;
    currentY = output;
    currentdE_dY = dE_dYOut;
    theta_grads = cell(nnet.num_layers, 1);
    layers = [input, nnet.hid_layers];
    for i = nnet.num_layers - (1:nnet.num_layers) + 1
        currentdE_dZ = currentdE_dY .* currentY .* (1 - currentY);
        theta_grads{i} =  (currentdE_dZ * [layers{i}; ones(1, num_examples)]') / num_examples;
        currentdE_dY = nnet.theta{i}(:, 1:end-1)' * currentdE_dZ;
        currentY = layers{i};
    end
    gradients = theta_grads;
end

