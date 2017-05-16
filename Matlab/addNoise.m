function X_noise = addNoise(X)

    for k = 1:size(X,1)
        X_noise(k,:) = awgn(X(k,:),-1);
    end
end