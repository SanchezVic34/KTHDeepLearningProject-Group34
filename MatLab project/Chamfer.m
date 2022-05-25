function Chamf_dist = Chamfer(X, Y)
    % X and Y are clouds of dots matrix of size(nx, d) and (ny, d) with d
    % the dimension of a dot and nx, ny the number of dots in X and Y.
    % The Chamfer distance gives a measure of how close two dots clouds
    % are.
    
    [nx, d] = size(X);
    ny = length(Y(1,:));
    Chamf_dist = 0;
    if isempty(X)
        X = [0 0 0 0];
    end
    if isempty(Y)
        Y = [0 0 0 0];
    end
    
    for n = 1:nx
        x = X(n,:);
        Chamf_dist = Chamf_dist + min(sum((Y-x).^2, 2));
    end
    for n = 1:ny
        y = Y(n,:);
        Chamf_dist = Chamf_dist + min(sum((X-y).^2, 2));
    end
    
end