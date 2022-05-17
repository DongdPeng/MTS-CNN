Nets={'drug_ATC1','drug_ATC2','drug_ATC3','drug_path','drug_side','drug_target','drug_tran','drug_enzyme'};
for i = 1 : length(Nets)
	tic
  	inputID = char(strcat('../matrix/', Nets(i), '.txt'));
	M = load(inputID);
	Sim = 1 - pdist(sidematrix, 'jaccard');
	Sim = squareform(Sim);

	Sim = Sim + eye(size(sidematrix,1));
	Sim(isnan(Sim)) = 0;      
    [m,n]=size(Sim);
    
	outputID = char(strcat('../Sim_', Nets(i), '.txt'));
	dlmwrite(outputID, Sim, '\t');
	toc
end

  
