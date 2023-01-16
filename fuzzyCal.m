function result = fuzzyCal(ss,cr,sf,sfp,fm)
fis = readfis('feature');
result = evalfis(fis,[ss,cr,sf,sfp,fm]);
end