function [cartesianMat, polarMat] = reconstruct(fullMatrix)

	K = [-100 0 200; 0 -100 200; 0 0 1];
	Mextleft = [0.707 0.707 0 -3; -0.707 0.707 0 -0.5; 0 0 1 3];
	Mextright = [0.866 -0.5 0 -3; 0.5 0.866 0 -0.5; 0 0 1 3];



	pts = [2 0 0; 3 0 0; 3 1 0; 2 1 0; 2 0 1; 3 0 1; 3 1 1; 2 1 1; 2.5 0.5 2];



	NN = 9;
	pix = zeros(NN,3);

	for i = 1:NN
		pixels = K*Mextleft * [pts(i,1) pts(i,2) pts(i,3) 1]';
		leftpix(i,:) = pixels./pixels(3);
		pixels = K*Mextright * [pts(i,1) pts(i,2) pts(i,3) 1]'
		rightpix(i,:) = pixels.pixels(3);
	end



	rightray = inv(K) * 


end