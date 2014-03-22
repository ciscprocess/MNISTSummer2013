function [ new_model ] = beam_search( nnet, inputs, targets, starting_temp )
%BEAM_SEARCH local 3-beam search with simulated annealing
%   Detailed explanation goes here
    temp1 = starting_temp;
    temp2 = starting_temp;
    temp3 = starting_temp;
    fprintf('Warning!! This function is DEPRECATED. It will not play nice with the other functions');
    beam1 = nnet;
    beam2 = nnet;
    beam3 = nnet;
    minerr = Inf;
    err = Inf;
    old_terr1 = Inf;
    old_terr2 = Inf;
    old_terr3 = Inf;
    minmod = 0;
    alpha = 0.01;
    moment_decay = 0.6;
    
    b1m1 = 0;
    b1m2 = 0;
    b1m3 = 0;
    
    b2m1 = 0;
    b2m2 = 0;
    b2m3 = 0;
    
    b3m1 = 0;
    b3m2 = 0;
    b3m3 = 0;
    
    for i = 1:10000
        b1grad = calc_gradient(beam1, inputs, targets);
        b2grad = calc_gradient(beam2, inputs, targets);
        b3grad = calc_gradient(beam3, inputs, targets);
        
        b1m1 = moment_decay * b1m1 - alpha * (b1grad.theta1 + randn(size(b1grad.theta1))*(temp1/(i)));
        b1m2 = moment_decay * b1m2 - alpha * (b1grad.theta2 + randn(size(b1grad.theta2))*(temp1/(i)));
        b1m3 = moment_decay * b1m3 - alpha * (b1grad.theta3 + randn(size(b1grad.theta3))*(temp1/(i)));

        b2m1 = moment_decay * b2m1 - alpha * (b2grad.theta1 + randn(size(b2grad.theta1))*(temp2/(i^1.02)));
        b2m2 = moment_decay * b2m2 - alpha * (b2grad.theta2 + randn(size(b2grad.theta2))*(temp2/(i^1.02)));
        b2m3 = moment_decay * b2m3 - alpha * (b2grad.theta3 + randn(size(b2grad.theta3))*(temp2/(i^1.02)));
        
        b3m1 = moment_decay * b3m1 - alpha * (b3grad.theta1 + randn(size(b3grad.theta1))*(temp3/(4*i)));
        b3m2 = moment_decay * b3m2 - alpha * (b3grad.theta2 + randn(size(b3grad.theta2))*(temp3/(4*i)));
        b3m3 = moment_decay * b3m3 - alpha * (b3grad.theta3 + randn(size(b3grad.theta3))*(temp3/(4*i)));
        
        beam1.theta1 = beam1.theta1 + b1m1;
        beam1.theta2 = beam1.theta2 + b1m2;
        beam1.theta3 = beam1.theta3 + b1m3;
        
        beam2.theta1 = beam2.theta1 + b2m1;
        beam2.theta2 = beam2.theta2 + b2m2;
        beam2.theta3 = beam2.theta3 + b2m3;
        
        beam3.theta1 = beam3.theta1 + b3m1;
        beam3.theta2 = beam3.theta2 + b3m2;
        beam3.theta3 = beam3.theta3 + b3m3;

        b1out = feed_forward(beam1, inputs);
        terr1 = calc_model_error(b1out, targets);
        
        b2out = feed_forward(beam2, inputs);
        terr2 = calc_model_error(b2out, targets);
        
        b3out = feed_forward(beam3, inputs);
        terr3 = calc_model_error(b3out, targets);
        
        if terr1 - old_terr1 > 0.5
            temp1 = temp1 + 2;
        end
        old_terr1 = terr1;
        
        if terr2 - old_terr2 > 0.5
            temp2 = temp2 + 2;
        end
        old_terr2 = terr2;
        
        if terr3 - old_terr3 > 0.5
            temp3 = temp3 + 2;
        end
        old_terr3 = terr3;
        
        [err, index] = min([terr1 terr2 terr3]);
        if err < minerr
            minerr = err;
            mods = [beam1, beam2, beam3];
            minmod = mods(index);
        end
        
        fprintf('Iteration %d error: %f, min beam: %d \n', i, err, index);
    end
    
    finout = feed_forward(minmod, inputs);
    err = calc_model_error(finout, targets);
    fprintf('Beam search ended. Min err: %f \n', i, err);
    new_model = minmod;
end

