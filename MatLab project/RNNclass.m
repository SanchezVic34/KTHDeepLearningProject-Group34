classdef RNNclass
   properties
      b {mustBeNumeric}
      c {mustBeNumeric}
      U 
      V
      W
   end
   
   methods
       
        function obj = initialize(obj, K, m)
            obj.b = zeros(m,1);
            obj.c = zeros(K,1);
            obj.U = randn(m, K)*(6/(m+K))^(0.5);
            obj.W = randn(m, m)*(6/(m+m))^(0.5);
            obj.V = randn(K, m)*(6/(m+K))^(0.5);
        end
       
        function hot_seq = synth_seq(obj, x0, h0, n, T)
            if ~exist('mode', 'var')
                mode=1;
            end
            K = length(x0);
            hot_seq = zeros(K, n);
            for t=1:n
                a=obj.W*h0 + obj.U*x0 + obj.b;
                h0 = tanh(a);
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
        
        function [loss, h, a, p] = forward(obj, X, Y, h0)
            [~, n] = size(X);
            seq = zeros(1, n, 'int32');
            a = zeros(length(obj.b), n);
            h = zeros(length(h0), n);
            p = zeros(length(obj.c), n);
            loss=0;
            for t=1:n
                a(:,t) = obj.W*h0 + obj.U*X(:,t) + obj.b;
                h(:,t) = tanh(a(:,t));
                o = obj.V*h(:,t) + obj.c;
                p(:,t) = exp(o)/sum(exp(o));
                loss = loss - Y(:,t)'*log(p(:,t));
                h0 = h(:,t);
            end
        end
        
        function [grads] = backward(obj, X, Y, h, h0, a, p)
            % Initialize gradient object
            grads=struct(); grads.b = zeros(length(obj.b),1); grads.c = obj.c; 
            grads.U = obj.U; grads.V = obj.V; grads.W = obj.W;
            grad_h = h';
            grad_a = a';
            [m, tau] = size(h);
            
            % Begin gradient computation
            grad_o = -(Y-p)';
            grads.V = grad_o'*h';
            grads.c = sum(grad_o, 1)';
            
            % Recursively compute the gradients of h and a (transposed)
            grad_h(tau, :) = grad_o(tau, :)*obj.V;
            grad_a(tau, :) = grad_h(tau, :)*diag(1-tanh(a(:,tau)).^2);
            for t=(tau-1):-1:1
                grad_h(t, :) = grad_o(t,:)*obj.V + grad_a(t+1, :)*obj.W;
                grad_a(t, :) = grad_h(t, :)*diag(1-tanh(a(: ,t)).^2);
            end
            
            grads.W = grad_a(2:tau, :)'*h(:, 1:tau-1)' + grad_a(1, :)'*h0';
            grads.U = grad_a'*X';
            for t = 1:tau
                grads.b = grads.b + diag(1-h(:,t).^2)*grad_h(t,:)';
            end
            
            for f = fieldnames(grads)'
                grads.(f{1}) = max(min(grads.(f{1}), 5), -5); % To keep the gradients within a small range
            end
        end
        
   end
   
end


















