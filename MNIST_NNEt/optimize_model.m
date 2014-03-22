function [ new_model, final_error ] = optimize_model( nnet, inputs, targets )
%OPTIMIZE_MODEL Summary of this function goes here
%   nnet is a neural network model (a struct)
%   inputs is a matrix of inputs
    epochs = 6000;
    [~, num_examples] = size(inputs);
    batch_size = 767;
    sample_size = 767;
     
    
    idx = randperm(num_examples);
    sampling_targets = targets(:, idx(1:sample_size));
    sampling_inputs = inputs(:, idx(1:sample_size));
    
    outputs = feed_forward(nnet, sampling_inputs);
    err = calc_model_error(outputs, sampling_targets);
    
    fprintf('First Epoch %d: error of %f \n', 0, err)
    alpha = 0.003;
    
    
    moment_decay = 0.6;
    momenta = cell(nnet.num_layers);
    for i = 1:nnet.num_layers
        momenta{i} = 0;
    end
    min_model = 0;
    min_err = Inf;
    
    for i = 1:epochs

        idx = randperm(num_examples);
        batch_targets = targets(:, idx(1:batch_size));
        batch_inputs = inputs(:, idx(1:batch_size));
        
        for k = 1:50
            grads = calc_gradient(nnet, batch_inputs, batch_targets);
            for j = 1:nnet.num_layers
                momenta{j} = moment_decay * momenta{j} - alpha * grads{j};
                nnet.theta{j} = nnet.theta{j} + momenta{j};
            end
        end
        
        idx = randperm(num_examples);
        sampling_targets = targets(:, idx(1:sample_size));
        sampling_inputs = inputs(:, idx(1:sample_size));
        
        outputs = feed_forward(nnet, sampling_inputs);
        err = calc_model_error(outputs, sampling_targets);
        if (err < min_err)
            min_err = err;
            min_model = nnet;
        end
        fprintf('Epoch %d: error of %f \n', i, err)
    end
    
    new_model = min_model;
    final_error = min_err;
end

