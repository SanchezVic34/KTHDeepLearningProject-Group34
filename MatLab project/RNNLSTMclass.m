classdef RNNLSTMclass
   properties
      c
      V
      W
   end
   
   methods
       
        function obj = initialize(obj, K, d)
            obj.W = randn(4*d, d+K)*((6/(4*d+d*K))^(0.5));
            obj.V = randn(K, d)*((6/(K+d))^(0.5));
            obj.c = zeros(K, 1);
        end

        function hot_seq = synth_seq(obj, x0, h0, c0, n, T)
            % Synthesize a sequence of length n, starting with character
            % x0, hidden states h0 and c0. The temperature T can go from 0
            % to +inf, if it equals 0 the generation process is
            % deterministic and the character generation uses max(p)
            if ~exist('mode', 'var')
                mode=1;
            end
            K = length(x0);
            d = length(h0);
            hot_seq = zeros(K, n);
            for t=1:n
                z = obj.W*[x0; h0]; % h0 is updated so that it is always h(:,t-1)
                a = tanh(z(1:d));
                i = 1./(1+exp(-z(d+1:2*d))); %exp(z(d+1:2*d))/sum(exp(z(d+1:2*d)));
                f = 1./(1+exp(-z(2*d+1:3*d))); %exp(z(2*d+1:3*d))/sum(exp(z(2*d+1:3*d)));
                o = 1./(1+exp(-z(3*d+1:4*d))); %exp(z(3*d+1:4*d))/sum(exp(z(3*d+1:4*d)));
                c0 = i.*a + f.*c0; % c0 is always c(t-1)
                h0 = o.*tanh(c0);
                out = obj.V*h0 + obj.c;
                if T==0
                    p = exp(out)/sum(exp(out));
                    [~, ixs] = max(p);
                else
                    p = exp(out./T)/sum(exp(out./T));
                    cp = cumsum(p);
                    rn = rand;
                    ixs = find(cp-rn >0);
                end
                hot_seq(ixs(1), t) = 1;
                x0 = zeros(size(x0));
                x0(ixs(1))=1;
            end
        end

        function [loss, p, h, a_hat, a, i, f, o, c] = forward(obj, X, Y, h0, c0)
            [n, T] = size(X);
            d = length(h0);
            
            h = zeros(d, T);
            a_hat = zeros(d, T);
            a = zeros(d, T);
            i = zeros(d, T);
            f = zeros(d, T);
            o = zeros(d, T);
            c = zeros(d, T);
            
            loss=0;
            for t=1:T
                z = obj.W*[X(:,t); h0]; % h0 is updated so that it is always h(:,t-1)
                a_hat(:, t) = z(1:d);
                a(:,t) = tanh(z(1:d));
                i(:,t) = 1./(1+exp(-z(d+1:2*d))); %exp(z(d+1:2*d))/sum(exp(z(d+1:2*d)));
                f(:,t) = 1./(1+exp(-z(2*d+1:3*d))); %exp(z(2*d+1:3*d))/sum(exp(z(2*d+1:3*d)));
                o(:,t) = 1./(1+exp(-z(3*d+1:4*d))); %exp(z(3*d+1:4*d))/sum(exp(z(3*d+1:4*d)));
                
                c(:,t) = i(:,t).*a(:,t) + f(:,t).*c0; % c0 is always c(t-1)
                h(:,t) = o(:,t).*tanh(c(:,t));
                out = obj.V*h(:,t) + obj.c;
                p(:,t) = exp(out)/sum(exp(out));
                
                loss = loss - Y(:,t)'*log(p(:,t));
                h0 = h(:,t);
                c0 = c(:,t);
            end
        end
        
        function [grads] = backward(obj, X, Y, p, h, h0, a_hat, a, i, f, o, c, c0)
            [d, T] = size(h);
            [K, ~] = size(X);
            
            % Initialize gradient object
            grads=struct(); grads.W = zeros(size(obj.W));
            grad_h = zeros(size(h(:,T)));
            grad_c = zeros(size(c(:,T)));
            
            % Begin gradient computation
            grad_out = -(Y-p);
            grads.V = grad_out*h';
            grads.c = sum(grad_out, 2);
            
            % Initiate grad_h(T)
            %grad_h = obj.V'*grad_out(:, T);
            
            % Recursively compute the gradients
            for t=T:-1:2
                grad_h = grad_h + obj.V'*grad_out(:, t);
                grad_o = grad_h.*tanh(c(:, t));
                grad_c = grad_c + grad_h.*o(:,t).*(1-tanh(c(:,t)).^2);
                grad_i = grad_c.*a(:,t);
                grad_f = grad_c.*c(:,t-1);
                grad_a = grad_c.*i(:,t);
                grad_c = grad_c.*f(:,t); % Now grad_c holds the value for grad_c(t-1)
                grad_z = [grad_a.*(1-tanh(a_hat(:,t)).^2); grad_i.*i(:,t).*(1-i(:,t)); grad_f.*f(:,t).*(1-f(:,t)); grad_o.*o(:,t).*(1-o(:,t))];
                grad_I = obj.W'*grad_z;
                grad_h = grad_I(K+1:end); % Get the value for grad_h from grad_I
                grads.W = grads.W + grad_z*[X(:,t); h(:,t-1)]';
            end
            
            % The last step must be performed outside because values at t=0
            t=1;
            grad_h = grad_h + obj.V'*grad_out(:, t);
            grad_o = grad_h.*tanh(c(:, t));
            grad_c = grad_c + grad_h.*o(:,t).*(1-tanh(c(:,t)).^2);
            grad_i = grad_c.*a(:,t);
            grad_f = grad_c.*c0;
            grad_a = grad_c.*i(:,t);
            grad_z = [grad_a.*(1-tanh(a_hat(:,t)).^2); grad_i.*i(:,t).*(1-i(:,t)); grad_f.*f(:,t).*(1-f(:,t)); grad_o.*o(:,t).*(1-o(:,t))];
            grads.W = grads.W + grad_z*[X(:,t); h0]';
            
            % Clip the gradients to avoid explosions
            for f = fieldnames(grads)'
                grads.(f{1}) = max(min(grads.(f{1}), 5), -5); % To keep the gradients within a small range
            end
        end
        
   end
   
end
