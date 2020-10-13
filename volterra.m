function [xtt,yt,txtvt] = volterra(xt,lags,beta)

xtt=[]; yt = []; txtvt = '';
if exist('lags','var')
    if ~isempty(lags)
        numlags = max(max([lags{:}]))+1;
    else
        error('must specify lags');
    end
end

L = size(xt,2);
xtt = zeros(numlags,L);
for ll=0:numlags-1
    xtt(ll+1, ll + (1:L-ll) ) = xt(1:L-ll);
end

if exist('beta','var')
    if ~isempty(beta)
        yt = zeros(size(xt));
        txtvt='Volterra process y(t) =';
        for term = 1:length(beta)
            lagst = lags{term};
            if numel(lagst)==2
                toadd = xtt( 1+lagst(1),:) .* (xtt( 1+lagst(2),:));
                txtvt = sprintf('%s\n  + (%0.3f + i%0.3f) x(t-%d) x(t-%d)',txtvt, ...
                    real(beta(term)),imag(beta(term)),lags{term}(1),lags{term}(2));
            else
                toadd = xtt( 1+lagst(1),:);
                txtvt = sprintf('%s\n + (%0.3f + i%0.3f) x(t-%d)',txtvt, ...
                    real(beta(term)),imag(beta(term)),lags{term}(1));
            end
            yt = yt + beta(term) * toadd;
        end
    end
end
