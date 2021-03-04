function make_BER_plot(results_collected_file)
tmp = load(results_collected_file);

results = tmp.results;
sweep = tmp.sweep;
params = tmp.params;

if isfield(sweep, 'training_blocksize_syms')
    sweep.n_train_symbols = sweep.training_blocksize_syms;
end

if ~isfield(sweep, 'n_train_symbols')
    sweep.n_train_symbols = NaN;
end


figdir = fullfile(fileparts(results_collected_file), 'figs')
grid = make_sweep_grid(sweep);
tmp = fieldnames(grid);
tmp = tmp{1};
dims = size(grid.(tmp));


BER = reshape(results.BER, dims);
BERavg = nanmean(BER, length(size(BER)));
tmp = ones(size(dims));
tmp = mat2cell(tmp, 1, tmp);
BER_outage = nanmean((BER > 0) ./ (1 - isnan(BER(tmp{:},:))), length(size(BER)));

ndels = length(sweep.del_spread);
ndops = length(sweep.dop_spread);
ntargs = length(sweep.n_target_symbols);

for tapidx = 1:length(sweep.n_deltaps);
for trainidx = 1:length(sweep.n_train_symbols)
figure(10);%figure(1000*tapidx + trainidx)
clf
targidx = 1;
for diagidx = 1:length(sweep.diag_load_level_dB);
for delidx = 1:ndels
    for dopidx = 1:ndops; 
        subplot(ndops,ndels,delidx+ndels*(dopidx-1));
        cla
        hold on
        turnon_styles = {'s', 'o'};
        alg_styles = {'--', '-', ':', '-.'};
        colors = get(gca, 'ColorOrder');
        turnon_colors = {colors(1,:), colors(2,:)};
        
        for algidx = 1:2
            for turnidx = 1:2
                plot( sweep.JNR_dB,  squeeze(BER_outage(delidx, dopidx,:, tapidx, trainidx, targidx, algidx, turnidx, diagidx, :))', 'Marker', turnon_styles{turnidx}, 'LineStyle', alg_styles{algidx} , 'Color', turnon_colors{turnidx}); 
            end
        end
        line(params.chan_params.SNR_dB*ones(1,2), [0 1], 'Color', [.7 .7 .7], 'LineStyle', '--')
        text(params.chan_params.SNR_dB, 0.5, 'S', 'Color', [.7 .7 .7], 'HorizontalAlignment', 'right')
%         ylim([0 1]);
%         xlim([0 51]);
%         set(gca, 'XTick', 0:10:50)
%         set(gca, 'YTick', 10.^(-5:-1))
        
        if dopidx == 1
            title(sprintf('%.1f us', 1e6*sweep.del_spread(delidx)));
        end
        if delidx == 1
            ylabel(sprintf('%.1f Hz', sweep.dop_spread(dopidx)));
        end
%         
%         if delidx == 1 && dopidx == 1
%             legend(sprintf('%d dB JNR', sweep.JNR_dB(1)), num2str(sweep.JNR_dB(2)), num2str(sweep.JNR_dB(3)), num2str(sweep.JNR_dB(4)));
%         end
%         

        if dopidx == length(sweep.dop_spread)
            xlabel('JNR (dB)');
        end

         set(gca, 'XGrid', 'on');
         set(gca, 'YGrid', 'on')
         set(gca, 'YMinorGrid', 'off')
    end
    
end
suplabel(sprintf('%d taps; %d dB diag. load', sweep.n_deltaps(tapidx), sweep.diag_load_level_dB(diagidx)), 't')
boldify
set(gcf, 'Color', 'white')
set(gcf, 'Position', [1 35 1600 1046]);
drawnow
export_fig(fullfile(figdir, sprintf('%d_taps_%d_trainsyms_%d_dB_diagload.png', sweep.n_deltaps(tapidx), sweep.n_train_symbols(trainidx), sweep.diag_load_level_dB(diagidx))));

end
end
end
% suplabel('BER','t');


% hleg = legend('5ms','10ms','15ms', '20ms', '25ms', 'Location', 'WestOutside');
% hleg = legend('10ms','20ms','30ms', '40ms', '50ms', 'Location', 'WestOutside');
% hleg.Title.String = 'Training Window';
% hleg.Position = [ 0.0119    0.8246    0.0755    0.0774];

% 
% nsnrs = length(sweep.SNR_dB);
% ntrains = length(sweep.n_train_symbols);
% figure(6)
% for snridx = 1:nsnrs; 
%     for trainidx = 2; %1:ntrains; 
% %         subplot(ntrains,nsnrs,snridx+nsnrs*(trainidx-1)); 
%         subplot(1,nsnrs,snridx); 
%         heatmap(sweep.dop_spread, sweep.del_spread*1e6, BER_1em2_outage(:,:, snridx, trainidx));
%     
%     
%         if trainidx == 1
%             title(sprintf('%d dB', sweep.SNR_dB(snridx)));
%         end
%         if snridx == 1
%             ylabel(sprintf('%.1f ms', floor(sweep.n_train_symbols(trainidx)/params.tx_params.Fs_symbol*1e3)));
%         end
% 
% 
%         if trainidx == length(sweep.n_train_symbols)
%             xlabel('Dop Spread (Hz)');
%         end
%     end
% end
%       
% suplabel('Outage Probability (>1% BER)','t');
% % boldify
% set(gcf, 'Color', 'white')
