%OMP Recovery
%  s-??????T-??????????N-????????; k: Sparsity
function hat_y=omp(s,T,N,k,eps)

Size=size(T);                                     %  ????????????
M=Size(1);                                        %  ????
hat_y=zeros(1,N);                                 %  ????????????(??????)????                     
Aug_t=[];                                         %  ????????(??????????????)
r_n=s;                                            %  ??????

for times=1:k;                                    %  ????????(??????????????1/4)
    for col=1:N;                                  %  ????????????????????
        product(col)=abs(T(:,col)'*r_n);          %  ????????????????????????????????(??????)
    end
    [val,pos]=max(product);                       %  ??????????????????????
    Aug_t=[Aug_t,T(:,pos)];                       %  ????????
    T(:,pos)=zeros(M,1);                          %  ??????????????????????????????????????????????????
    aug_y=pinv(Aug_t'*Aug_t)*Aug_t'*s;           %  ????????,?????????? modified by strin to pinv for numerical stability.
    r_n=s-Aug_t*aug_y;                            %  ????
    pos_array(times)=pos;                         %  ??????????????????????
    
    if (norm(r_n)<eps)                              %  ??????????
        break;
    end
end
hat_y(pos_array)=aug_y;                           %  ??????????
