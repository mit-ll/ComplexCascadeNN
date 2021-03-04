function results = collect_results(datadir)
% function results = collect_results(datadir, nchannels)

files = dir(fullfile(datadir, 'workspaces/results*.mat'));

tmp = load(fullfile(files(1).folder, files(1).name));
sweep = tmp.sweep;
params = tmp.params;

sweepvars = fieldnames(sweep);
numsweep = 1;
for idx = 1:length(sweepvars)
    numsweep = numsweep*numel(sweep.(sweepvars{idx}));
end

results.BER = NaN(1, numsweep);
results.BER_raw = NaN(1, numsweep);
results.start_block_idx = NaN(1,numsweep);
results.truth_start_block_idx = NaN(1,numsweep);

for fidx = 1:length(files)
    fprintf('%d/%d...', fidx, length(files));
    ridx = sscanf(files(fidx).name, 'results%d.mat');
    tmp = load(fullfile(files(fidx).folder, files(fidx).name));
    try
    results.BER(ridx) = tmp.results.BER;
    results.BER_raw(ridx) = tmp.results.BER_raw;
    results.start_block_idx(ridx) = tmp.results.start_block_idx;
    results.truth_start_block_idx(ridx) = tmp.results.truth_start_block_idx;
%     results.det_stats{ridx} = tmp.results.det_stats;
%     results.remod_metrics{ridx} = tmp.results.remod_metrics;
%     results.is_using_sync_weights{ridx} = tmp.results.is_using_sync_weights;
    results.start_block_idx(ridx) = tmp.results.start_block_idx;
    catch ME
        if ~isfield(tmp.results, 'ME')
            throw(ME);
        else
            warning(sprintf('file: %d\nerror: %s', fidx, ME.message));
        end
    end
end

save(fullfile(datadir, 'results_collected.mat'), 'results', 'sweep', 'params');
fprintf('\n')
