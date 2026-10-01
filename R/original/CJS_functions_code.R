#the rows of x correspond to the dimension d of the vector in all the functions below 

#standardise to margins_to_frechet
margins_to_frechet<-function(x){
  n<-dim(x)[2]
  d<-dim(x)[1]
  m<-matrix(0, nrow=d, ncol=n)
  for (i in 1:d){            
    m2<-ecdf(x[i, ])
    m[i,]<-(-log(pmin(n/(n+1)*(m2)(x[i,]))))^(-1/2)
  }
  return(m)
}

# computes squared angular components, used in the computation of the A matrix below, 
#and is based on Klüppelberg and Krali 2021.

#k corresponds to the number of radial threshold exceedances
omega_A<-function(x, k){
  
  c_sum<-colSums(rbind(x, 0)^2)
  
  #choose k largest radii
  I<-which(c_sum>=sort(c_sum, T)[k])
  
  #compute squared angular components
  w<-apply(rbind(x, 0)[, I], 2, function(x)x^2/sum(x^2))
  
  return(w)
}

#computes the scaling of the maximum M_I taken over the indices in I of the vector X.
sigma_M_I<-function(x, I, k){
  
  y<-omega_A(x[I,], k)
  
  #mass of angular measure, corresponding to dimension of I
  #the second argument subtracts 1 because the output of omega_A contains 
  #zeros in the last rows
  m<-ifelse(min(I)<0, length(y[,1])-1, length(I))
  
  #compute the squared scaling
  p<-m*mean(apply(rbind(y,-1), 2, function(x)max(x)))
  return(p)
}



#x is the stanrdadised data with Frechet 2 margins
#k is the number of threshold exceedances
#generations is a list object that contains the various generations
A_generations<-function(x, k , generations){
  
  d<-dim(x)[1]
  b=matrix(0, nrow= d, ncol= d)
  
  #rename elements in generations
  steps<-length(generations)
  aggr<-vector()
  for(g in c(1:steps)){
    aggr[g]<-length(generations[[g]])
    if(g==1){
      generations[[g]]<-c((d-aggr[1]+1):d)
    }else{
      generations[[g]]<-c((d-sum(aggr[1:g])+1):(d-sum(aggr[1:(g-1)])))
    }
  }
  
  
  for (i in d:2){
    b[i,i]<-ifelse(i<=(d-1), sigma_M_I(x, c(i:d), k), sigma_M_I(x, d, k))-ifelse(i<=(d-1), ifelse(i<=(d-2), sigma_M_I(x, c((i+1):d), k), sigma_M_I(x, d, k)),0)
  }
  b[1,1]<-sigma_M_I(x,c(1:d), k)-sum(diag(b))
  
  for (i in 1:(d-1)){
    for (j in (i+1):(d)){
      if(i==1){
        b[i,j]<-sigma_M_I(x,-c((i+1):j), k)-sum(b[i,c(i:(j-1))])-ifelse(j<d,sigma_M_I(x,-c(i:(j)), k),0); 
      }
      else{  
        b[i,j]<-sigma_M_I(x,-c((1:(i-1)),((i+1):j)), k)-sum(b[i,c(i:(j-1))])-ifelse(j<d,sigma_M_I(x,-c(1:j), k),0); 
      }
      b2<-b;
      #set to zero the path weights of nodes in the same generation
      for(g in c(1:steps)){
        b[generations[[g]],generations[[g]]]<-0;
      } 
      diag(b)<-diag(b2)
    } 
  }
  return(b)
}

#returns the matrix A after removing all paths which are not max-weighted. Corresponds to
#a weighted adjacency matrix for the DAG D^A_{\epsilon}. We use eps instead of \delta here
mwp_gr_eps<-function(P_orig, eps){
  #starting matrix A
  P<-P_orig; 
  d<-dim(P)[1]; 
  midpath<-rep(0,d)
  for(m in 1:d){
    for(i in c(1:d)[-m]){      midpath<-rep(0,d)
    for(k in c(1:d)[-c(i,m)]){
      #compute path weight crossing through k which is a descendant of m and ancestor of i
      midpath[k]<-ifelse((P_orig[i,k]*P_orig[k,m])>0, P_orig[i,k]*P_orig[k,m]/P_orig[k,k], 0)
    }; 
    #apply the hard thresholding
    if((max(midpath,na.rm=T)+eps)>=P_orig[i,m]){
      #define the matrix P, representing the "new" matrix A which corresponds to the minimal DAG
      P[i,m]<-0
    }else{P[i,m]<-P_orig[i,m]}
    
    }
  }
  return(P)
}




#computes the  squared omega from the angular measures of x
#the vector x here contains only those margins over which we
#compute the scalings
omega_caus_order<-function(x, k){
  y<-colSums(x^2)
  so<-sort(y,T)[k]
  ind<-which(y>=so)
  w<-(abs(x)^2/y[col(x)])[,ind]
  return(w)
}

#computes the scaling sigma_M^2 using the angular measure of x only
sigma_M<-function(x, k){
  
  y<-omega_caus_order(x,k)
  p<-2*sum(colMaxs(y))/k
  return(p)
}


#funtion which computes initial differences of the scalings in Agorithm 1
pair_sigma_diff<-function(z, a, k){
  v<-sigma_M(rbind(z), k)
  q<-(1+a^2)/2*sigma_M(c(a,1)*z, k)-v-(a^2-1)
  
  return(q)
}

#computes the scaling of partially rescaled maxima. 
#Here we leave unscaled the first component
#in I, and then scale by "a", the remaining components.
#ijI should be ordered as c(i, j, I)
sigma_i_aj_aI_diff<-function(x, ijI, k, a){
  
  y<-rbind(x[ijI[1],],a*x[ijI[-1],])
  d<-dim(y)[1]
  dim_jI<-length(ijI)-1
  dim_i<- d-dim_jI
  y<-omega_caus_order(y, k)
  #define vector with angular components, where the ones
  #corresponding to j and I are divided by the scalars a^2
  y3<-rbind(y[1,],y[-1,]/a^2);
  
  #the mass of the angular measure of y
  mass<-(dim_i+a^2*dim_jI)
  
  #sigma_i_aj_aI^2
  p<-mass*sum(colMaxs(y))/k
  
  #take maximum of the squared angular components 
  #indexed in j and I
  ycm<-colMaxs(y3[-1,])
  
  #take difference of squared scalings
  #second summand corresponds to sigma_i_j_I^2
  #third summand corresponds to sigma_aj_aI^2
  pdif<-p-mass*sum(colMaxs(rbind(y3[1,],ycm)))/k-(a^2-1)*(mass)*sum(ycm)/k
  return(pdif)
}


#this computes the causal ordering, similar to Krali 2025.
causord_eps_a<-function(x, a, k, eps){
  d<-dim(x)[1]
  Delta<-matrix(0,d,d)
  generations<-list()
  
  for(i in 1:d){
    for(j in 1:d){Delta[j,i]<-pair_sigma_diff(x[c(i,j),], a, k)
    Delta[i,i]<-Inf
    }
  }
  
  #initial step of Algorithm 1 of Krali 2025
  Delta_col_Min<-apply(Delta,2,function(x)min(x))
  id<-which(abs(Delta_col_Min-max(Delta_col_Min))<=abs(eps*max(Delta_col_Min)));
  if(length(id)>1){
    id<-id[sort(Delta_col_Min[id],decreasing=F,index.return=T)$ix]
  }else{
    id<-id
  }
  #first generation with initial nodes
  generations[[1]]<-id
  g<-2
  while(sum(id>0)<=(d-2)){
    id1<-id
    Delta<-matrix(0,d,d);
    for(i in c(1:d)[-id]){
      for(j in c(1:d)[-id]){
        Delta[i,j]<-sigma_i_aj_aI_diff(x, c(i,j,id),  k, a)
      }
      Delta[i,i]<-Inf
    }
    Delta[id,]<- Inf; Delta[,id]<-Inf 
    
    #first take colMins over matrix Delta to preserve indices
    Delta_col_Min2<-apply(Delta, 2, function(x)min(x)); 
    
    #then find the colMins of Delta after removing the indices in I
    Delta_col_Min3<-apply(Delta[c(1:d)[-id],c(1:d)[-id]], 2, function(x)min(x))
    
    #lines 4 and 5 in Krali 2025
    id<-which(abs(Delta_col_Min2-max(Delta_col_Min3))<=eps*abs(max(Delta_col_Min3))); 
    #sort indices which are within error margins of largest column minimum
    if(length(id)>1){
      id<-c(id[sort(Delta_col_Min2[id],decreasing=F, index.return=T)$ix], id1)
    }else{
      id<-c(id, id1)
    }
    generations[[g]]<-id[-which(id%in%id1)]
    id<-id; 
    g<-g+1
    #remove the following comment to see the generations, i.e., the set id updates with the generations
    #print(id)
    
  } 
  generations[[g]]<-c(1:d)[-which(c(1:d)%in%id)]
  id<-c(c(1:d)[-which(c(1:d)%in%id)], id);
  
  output<-list(id, generations)
  names(output)<-c("I", "generations")
  return(output)
}