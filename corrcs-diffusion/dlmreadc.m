%%% dlmreadc 
%% this function is a reminscent of dlmread, except that it support ASCII strings and returns a cell matrix.
function res = dlmreadc(filename, dlm)
    fid=fopen(filename);
    res = [];
    li = 1;
    while true
            tline = fgetl(fid);
            if tline == -1, break, end
            pos = findstr(tline, dlm);
            pos = [0 pos length(tline)+1];
            for i = 1:length(pos)-1
                res{li, i} = tline(pos(i)+1:pos(i+1)-1);
                res_num = str2num(res{li, i});
                if ~isempty(res_num)
                    res{li, i} = res_num;
                end
            end
            li = li+1;
     end
end