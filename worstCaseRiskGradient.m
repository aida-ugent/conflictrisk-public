function wcrG = worstCaseRiskGradient(ind, L, S)
    n = size(L,2);
    I = eye(n);
    if ind == 1
        wcrG = L*inv(L+I)^2*S*S'*inv(L+I) + inv(L+I)*S*S'*inv(L+I)^2*L;
    elseif ind == 2
        wcrG = inv(L+I)^2*S*S'*inv(L+I)^2 - L*inv(L+I)^2*S*S'*inv(L+I)^2*L;
    elseif ind == 3
        wcrG = -inv(L+I)*S*S'*inv(L+I)^2 - inv(L+I)^2*S*S'*inv(L+I);
    else
        wcrG = -inv(L+I)*S*S'*inv(L+I);
    end
%     M = (M+M')/2;
end