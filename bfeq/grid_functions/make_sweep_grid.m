function grid = make_sweep_grid(sweep)

sweepcell = struct2cell(sweep);
sweepfields = fieldnames(sweep);

gridcell = cell(1, length(sweepcell));

[gridcell{:}] = ndgrid(sweepcell{:});

grid = cell2struct(gridcell, sweepfields, 2);