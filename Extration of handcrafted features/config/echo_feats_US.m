function [x,feats] = echo_feats_US(I,BW)

[x_hist,feats_hist] = histfeatures(I,BW);
[x_cor,feat_cor] = autocorr(I,BW);
[x_av,feats_av] = avmass(I,BW);
[x_lbp,feats_lbp] = bgc(I,BW,'lbp');
[x_clx,feats_clx] = clxcurve(I,BW);
[x_fra,feats_fra] = fractaltexture(I,BW);
D = [1 2 4 8]; % Distances in pixels
[x_glcm,feats_glcm] = glcm(I,BW,64,D,1,'mean');
[x_law,feats_law] = lawsenergy(I,BW,1);
[x_rcm,feats_rcm] = rcm(I,BW);
[x_nrg,feat_nrg] = nrg(I,BW);
x = [x_hist x_cor x_av  x_lbp x_clx x_fra x_glcm x_law x_rcm x_nrg];
feats = [feats_hist, feat_cor, feats_av, feats_lbp, feats_clx, feats_fra, feats_glcm, feats_law, feats_rcm, feat_nrg];
end