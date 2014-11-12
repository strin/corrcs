%% raw data.
pollution = dlmreadc('pm25/data_augmented.csv', ',');
station = csvread('pm25/station.csv');
n = size(station, 1);
%%

pulltion_mat = zeros(size(pollution));
for i = 1:size(pollution,1)
    i
    for j = 1:size(pollution, 2)
        try
            pollution_mat(i,j) = cell2mat(pollution(i,j));
        catch err
            pollution_mat(i,j) = NaN;
            % ignore.
        end
    end
end
%%
pollutionx = [pollution_mat(:,1) pollution_mat(:,3) pollution_mat(:,4) pollution_mat(:,5) pollution_mat(:,6)];


%% extract common time period.
date = unique(pollutionx(:,end));
for ni = 1:n
    date = intersect(date, pollutionx(pollutionx(:,1)==1000+ni & ~isnan(pollutionx(:,2)) & ~isnan(pollutionx(:,3)) & ~isnan(pollutionx(:,4)),end));
end
K = length(date);
clear feat;
feat.pm25 = zeros(K, n);
feat.pm10 = zeros(K, n);
feat.no2 = zeros(K, n);
for ni = 1:n
    data_ni = pollutionx(pollutionx(:,1) == 1000+ni, :);
    [tmp,IA,IB] = intersect(data_ni(:,end), date);
    feat.pm25(:,ni) = data_ni(IA, 2);
    feat.pm10(:,ni) = data_ni(IA, 3);
    feat.no2(:,ni) = data_ni(IA, 4);
end