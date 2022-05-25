function [RNNstar, Loss, cf, hf] = AdaGradLSTM(RNN, onehot_data, seq_length, eta, n_epochs, T_mes)

    %% Initialization
    [K, n_chars] = size(onehot_data);
    if ~exist('T_mes', 'var')
        T_mes = 100;
    end
    t_mes=0;
    [~, d] = size(RNN.V);
    for f = fieldnames(RNN)'
        mg.(f{1}) = 0;
    end
    
    % Init measurements
    Loss = zeros(floor(n_chars/(seq_length*T_mes))+1,1);
    smooth_loss = 109; min_loss = 109;

    %% Loop through epochs
    for epoch = 1:n_epochs
        h0 = zeros(d, 1);
        c0 = zeros(d, 1);
        for e = 1:seq_length:n_chars-seq_length
            % Data extraction
            X = onehot_data(:, e:e+seq_length-1);
            Y = onehot_data(:, e+1:e+seq_length);
            
            % Gradient update
            [loss, p, h, a_hat, a, i, f, o, c] = RNN.forward(X, Y, h0, c0); % h0 updated at the end of that loop
            grads = RNN.backward(X, Y, p, h, h0, a_hat, a, i, f, o, c, c0);
            for f = fieldnames(grads)'
                mg.(f{1}) = mg.(f{1}) + grads.(f{1}).^2;
                RNN.(f{1}) = RNN.(f{1}) - eta./sqrt(mg.(f{1})+eps).*grads.(f{1});
            end
            
            % Store the loss every T_mes steps
            smooth_loss = 0.999*smooth_loss + 0.001*loss;
            t_mes=t_mes+1;
            if mod(t_mes,T_mes)==0
                fprintf("Update step %d/%d completed, smooth loss value: %f \n", t_mes, floor(n_chars/seq_length)*n_epochs, smooth_loss)
                Loss(t_mes/T_mes) = smooth_loss;
                if smooth_loss<min_loss
                    RNNstar=RNN;
                    min_loss=smooth_loss;
                    cf=c0;
                    hf=h0;
                end
            end
            h0 = h(:,end);
            c0 = c(:,end);
        end
        
    end
end