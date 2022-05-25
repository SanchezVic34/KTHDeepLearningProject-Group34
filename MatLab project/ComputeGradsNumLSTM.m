function num_grads = ComputeGradsNumLSTM(X, Y, RNN, hnum, h0, c0)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, hnum, h0, c0);
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, hnum, h0, c0)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - hnum;
    l1 = RNN_try.forward(X, Y, h0, c0);
    RNN_try.(f)(i) = RNN.(f)(i) + hnum;
    l2 = RNN_try.forward(X, Y, h0, c0);
    grad(i) = (l2-l1)/(2*hnum);
end

