function [] = visualizeRows(mat)
  for i = 1:size(mat, 1)
    plot(mat(i,:));
    pause;
  end
end