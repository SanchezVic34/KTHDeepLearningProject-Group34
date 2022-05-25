function [RNNstar, Loss, hf] = AdaGrad(RNN, onehot_data, seq_length, eta, n_epochs, T_mes, dodisp)

    %% Initialization
    [K, n_chars] = size(onehot_data);
    if ~exist('T_mes', 'var')
        T_mes = 100;
    end
    t_mes=0;
    m = length(RNN.b);
    for f = fieldnames(RNN)'
        mg.(f{1}) = 0;
    end
    
    % Init measurements
    RNNstar=RNN;
    Loss = zeros(floor(n_chars/(seq_length*T_mes))+1,1);
    smooth_loss = 109; min_loss = 109;
    
    h0 = zeros(m, 1); hf=h0;
    %% Loop through epochs
    for epoch = 1:n_epochs
        hf = h0;
        for e = 1:seq_length:n_chars-seq_length
            % Data extraction
            X = onehot_data(:, e:e+seq_length-1);
            Y = onehot_data(:, e+1:e+seq_length);
            
            % Gradient update
            [loss, h, a, p] = RNN.forward(X, Y, h0); % h0 updated at the end of that loop
            grads = RNN.backward(X, Y, h, h0, a, p);
            for f = fieldnames(grads)'
                mg.(f{1}) = mg.(f{1}) + grads.(f{1}).^2;
                RNN.(f{1}) = RNN.(f{1}) - eta./sqrt(mg.(f{1})+eps).*grads.(f{1});
            end
            
            % Store the loss every T_mes steps
            smooth_loss = 0.999*smooth_loss + 0.001*loss;
            t_mes=t_mes+1;
            if mod(t_mes,T_mes)==0
                if dodisp
                    fprintf("Update step %d/%d completed, smooth loss value: %f \n", t_mes, floor(n_chars/seq_length)*n_epochs, smooth_loss)
                end
                Loss(t_mes/T_mes) = smooth_loss;
                if smooth_loss<min_loss
                    RNNstar=RNN;
                    min_loss=smooth_loss;
                end
            end
            h0 = h(:,end);
        end
        
    end
end