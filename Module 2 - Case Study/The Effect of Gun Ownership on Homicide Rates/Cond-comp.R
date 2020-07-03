###########################################################################################
#  Program:  Functions for estimating moments for using Machine Learning Methods.         #
#  Reference:  "Double Machine Learning for Causal and Treatment Effects",  MIT WP 2016   #
#  by V.Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey           #
#  These are preliminary programs that implement a 2-way split version of the estimators. #
###########################################################################################

source("Functions.R") 

DoubleML <- function(data, y, d, xx, xL, method, K, est, ite=1){
  
  TE        <- matrix(0,ite,(length(method)+1))
  STE       <- matrix(0,ite,(length(method)+1))
  result    <- matrix(0,2,(length(method)+1))
  MSE1      <- matrix(0,length(method),K)
  MSE2      <- matrix(0,length(method),K)
  MSE3      <- matrix(0,length(method),K)
  cond.comp <-matrix(list(),length(method),K)
  
  binary  <-  as.numeric(checkBinary(data[,d]))

  for(i in 1:ite){
  
    county    = unique(data$CountyCode)
    split     = runif(length(county))
    citygroup = as.numeric(cut(split,quantile(split,probs = seq(0, 1, 1/K)),include.lowest = TRUE))  # groups for K-fold cv
    ii        = county[citygroup == 1]
    cvgroup   = rep(2, nrow(data))
    cvgroup[data$CountyCode %in% ii ] = 1
    
    for(k in 1:length(method)){   
      
      cat(method[k],'\n')
      
      if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet", "Lava", "PLava")==method[k])){
        x=xL
      } else {
        x=xx
      }
      
      for(j in 1:K){   
        
        cat('  split',j,'\n')
        
        ii = cvgroup == j
        nii = cvgroup != j
        
        datause = as.data.frame(data[nii,])
        dataout = as.data.frame(data[ii,])  
        
        if(est=="interactive"){
          cond.comp[[k,j]]         <-cond_comp(datause=datause, dataout=dataout, y, d, x, method[k], linear=0, xL, binary);
          
          MSE1[k,j]               <- cond.comp[[k,j]]$err.yz0
          MSE2[k,j]               <- cond.comp[[k,j]]$err.yz1
          MSE3[k,j]               <- cond.comp[[k,j]]$err.z
          
          drop                   <- which(cond.comp[[k,j]]$mz_x>0.01 & cond.comp[[k,j]]$mz_x<0.99)      
          mz_x                   <- cond.comp[[k,j]]$mz_x[drop]
          my_z1x                 <- cond.comp[[k,j]]$my_z1x[drop]
          my_z0x                 <- cond.comp[[k,j]]$my_z0x[drop]
          yout                   <- dataout[drop,y]
          dout                   <- dataout[drop,d]
          TE[i,k]                <- ATE(yout, dout, my_z1x, my_z0x, mz_x)/K + TE[i,k];
          STE[i,k]               <- (1/(K^2))*((SE.ATE(yout, dout, my_z1x, my_z0x, mz_x))^2) + STE[i,k];
        }
        
        if(est=="plinear"){

          cond.comp[[k,j]]         <- cond_comp(datause=datause, dataout=dataout, y, d, x, method[k], linear=1, xL, binary);
          MSE1[k,j]                    <- cond.comp[[k,j]]$err.y
          MSE2[k,j]                    <- cond.comp[[k,j]]$err.z
          
          lm.fit.ry              <- lm(as.matrix(cond.comp[[k,j]]$ry) ~ as.matrix(cond.comp[[k,j]]$rz)-1);
          ate                    <- lm.fit.ry$coef;
          HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');
          STE[i,k]               <- (1/(K^2))*(diag(HCV.coefs)) +  STE[i,k] 
          if(method[k]=="Forest"){STE[i,k]             <-  (1/(K^2))*((summary(lm.fit.ry)$coef[2])^2) + STE[i,k]}
          
          TE[i,k]               <- ate/K + TE[i,k] ;

        }
      }  
    }
  }  
  
  if(est=="interactive"){
  
    p <- which.min(rowMeans(MSE1))
    l <- which.min(rowMeans(MSE2))
    m <- which.min(rowMeans(MSE3))
    
    cat('  best method for E[Y|X, D=0]:',method[p],'\n')
    cat('  best method for E[Y|X, D=1]:',method[l],'\n')
    cat('  best method for E[D|X]:',method[m],'\n')
  
  }
  
  if(est=="plinear"){
    
    p <- which.min(rowMeans(MSE1))
    l <- which.min(rowMeans(MSE2))
    
    cat('  best method for E[Y|X]:',method[p],'\n')
    cat('  best method for E[D|X]:',method[l],'\n')
    
  }

  for(j in 1:K){  
    
    ii = cvgroup == j
    nii = cvgroup != j
    
    datause = as.data.frame(data[nii,])
    dataout = as.data.frame(data[ii,])  
    
    if(est=="interactive"){
    
      drop                   <- which(cond.comp[[m,j]]$mz_x>0.01 & cond.comp[[m,j]]$mz_x<0.99)      
      mz_x                   <- cond.comp[[m,j]]$mz_x[drop]
      my_z1x                 <- cond.comp[[l,j]]$my_z1x[drop]
      my_z0x                 <- cond.comp[[p,j]]$my_z0x[drop]
      yout                   <- dataout[drop,y]
      dout                   <- dataout[drop,d]
      
      TE[i,(k+1)]                <- ATE(yout, dout, my_z1x, my_z0x, mz_x)/K + TE[i,(k+1)];
      STE[i,(k+1)]               <- (1/(K^2))*((SE.ATE(yout, dout, my_z1x, my_z0x, mz_x))^2) + STE[i,(k+1)];
    
    }
    
    if(est=="plinear"){
      
      lm.fit.ry              <- lm(as.matrix(cond.comp[[p,j]]$ry) ~ as.matrix(cond.comp[[l,j]]$rz)-1);
      ate                    <- lm.fit.ry$coef;
      HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');
      STE[i,(k+1)]           <- (1/(K^2))*(diag(HCV.coefs)) +  STE[i,(k+1)] 
      if(method[p]=="Forest" || method[l]=="Forest"){STE[i,(k+1)]             <-  (1/(K^2))*((summary(lm.fit.ry)$coef[2])^2) + STE[i,(k+1)]}
      
      TE[i,(k+1)]               <- ate/K + TE[i,(k+1)] ;
      
    }
  }
  
  colnames(result)   <- c(method, "best") 
  rownames(MSE1)   <- c(method) 
  rownames(MSE2)   <- c(method) 
  rownames(MSE3)   <- c(method) 
  rownames(result)   <- c("ATE", "se")
  result[1,] <- colMeans(TE)
  result[2,] <- sqrt(STE)
  
  return(result)
}  

cond_comp <- function(datause, dataout, y, d, x, method, linear,xL, binary){
  
  form_y   = y
  form_d   = d
  form_x   = x
  form_xL  = xL
  ind_u   = which(datause[,d]==1)
  ind_o   = which(dataout[,d]==1)
  err.yz1 = NULL
  err.yz0 = NULL
  my_z1x  = NULL
  my_z0x  = NULL
  
  ########################## Boosted  Trees ###################################################;
  
  if(method=="Boosting")
  {
    
    if(linear==0){
      
      fit            <- boost(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, distribution='gaussian')
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, n.trees=fit$best, dataout, type="response") 
      
      fit            <- boost(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, distribution='gaussian')
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model, n.trees=fit$best, dataout, type="response") 
      
    }
    
    if(binary==1){
      fit            <- boost(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, distribution='adaboost')
      mis.z          <- error(fit$yhatout, dataout[,d])$mis
    }
    
    if(binary==0){
      fit            <- boost(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, distribution='gaussian')
      mis.z          <- NA
    }
    
      err.z          <- error(fit$yhatout, dataout[,d])$err
      mz_x           <- fit$yhatout       
      rz             <- fit$resout 
      err.z          <- error(fit$yhatout, dataout[,d])$err 

    
    fit            <- boost(datause=datause, dataout=dataout,  form_x, form_y)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err
    
  }  
  
  
  ########################## Neural Network(Nnet Package) ###################################################;   
  
  
  if(method=="Nnet"){
    
    if(linear==0){
      
      fit            <- nnetF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      dataouts       <- as.data.frame(scale(dataout, center = fit$min, scale = fit$max - fit$min))
      my_z1x         <- predict(fit$model, dataouts)*(fit$max[fit$k]-fit$min[fit$k])+fit$min[fit$k] 
      
      fit            <- nnetF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      dataouts       <- as.data.frame(scale(dataout, center = fit$min, scale = fit$max - fit$min))
      my_z0x         <- predict(fit$model, dataouts)*(fit$max[fit$k]-fit$min[fit$k])+fit$min[fit$k] 
    }
    
    if(binary==1){
      fit            <- nnetF(datause=datause, dataout=dataout, form_x=form_x, form_y=form_d, clas=TRUE)
      mis.z          <- error(fit$yhatout, dataout[,d])$mis
    }
    
    if(binary==0){
      fit            <- nnetF(datause=datause, dataout=dataout, form_x=form_x, form_y=form_d, clas=FALSE)
      mis.z          <- NA
    }
    
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mz_x           <- fit$yhatout       
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- nnetF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err    
    
  } 
  
  ########################## Lasso and Post Lasso(Hdm Package) ###################################################;    
  
  if(method=="RLasso" || method=="PostRLasso"){
    
    post = FALSE
    if(method=="PostRLasso"){ post=TRUE }
    
    if(linear==0){
      
      fit            <- rlassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_y, post)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, newdata=formC(form_y, form_x, dataout)$x , type="response") 
      
      fit            <- rlassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_y, post)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model, newdata=formC(form_y, form_x, dataout)$x, type="response")   
      
    }
    
    if(binary==1){
      fit            <- rlassoF(datause=datause, dataout=dataout,  form_x, form_d, post, logit=TRUE)
      mis.z          <- error(fit$yhatout, dataout[,d])$mis
    }
    
    if(binary==0){
      fit            <- rlassoF(datause=datause, dataout=dataout,  form_x, form_d, post, logit=FALSE)
      mis.z          <- NA
    }   
    
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mz_x           <- fit$yhatout    
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- rlassoF(datause=datause, dataout=dataout,  form_x, form_y, post)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err            
  }    
  
  
  ########################## Lasso and Post Lasso(Hdm Package) ###################################################;    
  
  if(method=="Ridge" || method=="Lasso" || method=="Elnet"){
    
    if(method=="Ridge"){ alp=0 }
    if(method=="Lasso"){ alp=1 }
    if(method=="Elnet"){ alp=0.5 }
    
    if(linear==0){
      
      fit            <- lassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_y, alp=alp)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      fit.p          <- lm(form,  x = TRUE, y = TRUE, data=dataout);
      my_z1x         <- predict(fit$model, newx=fit.p$x[,-1] ) 
      
      fit            <- lassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_y, alp=alp)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      fit.p          <- lm(form,  x = TRUE, y = TRUE, data=dataout);
      my_z0x         <- predict(fit$model,  newx=fit.p$x[,-1])   
      
    }
    
    if(binary==1){
      fit            <- lassoF(datause=datause, dataout=dataout,  form_x, form_d, logit=TRUE, alp=alp)
      mis.z          <- error(fit$yhatout, dataout[,d])$mis
    }
    
    if(binary==0){
      fit            <- lassoF(datause=datause, dataout=dataout,  form_x, form_d, logit=FALSE, alp=alp)
      mis.z          <- NA
    }   
    
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mz_x           <- fit$yhatout    
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- lassoF(datause=datause, dataout=dataout,  form_x, form_y, alp=alp)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err            
  }    
  
  ############# Random Forest ###################################################;
  
  if(method=="Forest" | method=="TForest"){
    
    tune = FALSE
    if(method=="TForest"){tune=TRUE}
    
    
    if(linear==0){
      
      fit            <- RF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, nodesize=5, ntree=1000, tune=tune)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, dataout, type="response") 
      
      fit            <- RF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, nodesize=5, ntree=1000, tune=tune)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model, dataout, type="response")
      
    }
    
    if(binary==1){
      fit            <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=paste("as.factor(",form_d,")"), nodesize=1, ntree=1000, reg=FALSE, tune=tune)
      mis.z          <- error(as.numeric(fit$yhatout), dataout[,y])$mis
    }
    
    if(binary==0){
      fit            <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, nodesize=5, ntree=1000, reg=TRUE, tune=tune)
      mis.z          <- NA
    }   
    
    err.z          <- error(as.numeric(fit$yhatout), dataout[,y])$err
    mz_x           <- fit$yhatout       
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, nodesize=5, ntree=1000, tune=tune)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err      
  }
  
  ########################## Regression Trees ###################################################;     
  
  if(method=="Trees"){
    
    if(linear==0){
      
      fit            <- tree(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, dataout) 
      
      fit            <- tree(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model,dataout)   
      
    }
    
    fit            <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d)
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mis.z          <- error(fit$yhatout, dataout[,d])$mis
    mz_x           <- fit$yhatout       
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err   
  }
  
  
  ########################## Lava and Post Lava ###################################################;    
  
  if(method=="Lava" || method=="PLava")
  {
    
    post=FALSE
    if(method=="PLava"){post=TRUE}
    
    if(linear==0){
      
      fit            <- lava(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, post=post)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      scaleout       <- formC(form_y, form_x, dataout)$x[,-1] - matrix(1,nrow(dataout),1)%*% fit$mean
      my_z1x         <- scaleout %*% fit$model + fit$meanY
      
      fit            <- lava(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, post=post)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      scaleout       <- formC(form_y, form_x,dataout)$x[,-1] - matrix(1,nrow(dataout),1)%*% fit$mean
      my_z0x         <- scaleout %*% fit$model + fit$meanY
    
    }
    
    fit            <- lava(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, post=post)
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mis.z          <- error(fit$yhatout, dataout[,d])$mis
    mz_x           <- fit$yhatout     
    
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err      
    
    fit            <- lava(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, post=post)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err  
    
  }  
  
  
  if(method=="RLassoForest" || method=="PostRLassoForest"){
    
    post=FALSE
    tune=FALSE
    ntree = 5
    if(method=="PostRLassoForest"){post=TRUE}
  
    if(linear==0){
    
      rlasso          <- rlassoF(datause[ind_u,], dataout=dataout[ind_o,], form_xL, form_y, post, logit=FALSE)
      forest          <- RF(datause=NA, dataout=NA, x=formC(form_y, form_x,datause[ind_u,])$x[,-1], y=rlasso$resuse, xout=formC(form_y, form_x,dataout[ind_o,])$x[,-1], yout=rlasso$resout,  nodesize=5, ntree=ntree, tune=tune)
      my_z1x          <- predict(rlasso$model, newdata=formC(form_y, form_xL,dataout)$x, type = "response")  + predict(forest$model, newdata=formC(form_y, form_x,dataout)$x, type = "response")
      
      y1hat           <- rlasso$yhatout + forest$yhatout
      err.yz1         <- error(y1hat, dataout[ind_o,y])$err
      
      rlasso          <- rlassoF(datause[-ind_u,], dataout[-ind_o,], form_xL, form_y, post, logit=FALSE)
      forest          <- RF(datause=NA, dataout=NA, x=formC(form_y, form_x,datause[-ind_u,])$x[,-1], y=rlasso$resuse, xout=formC(form_y, form_x,dataout[-ind_o,])$x[,-1], yout=rlasso$resout  , form_y, form_x, nodesize=5, ntree=ntree, tune=tune)
      my_z0x          <- predict(rlasso$model, formC(form_y, form_xL,dataout)$x, type = "response")  + predict(forest$model, newdata=formC(form_y, form_x,dataout)$x, type = "response")
      
      y0hat           <- rlasso$yhatout + forest$yhatout
      err.yz0         <- error(y0hat, dataout[-ind_o,y])$err
    
    }
    
    rlasso          <- rlassoF(datause, dataout, form_xL, form_d, post, logit=TRUE)
    forest          <- RF(datause=NA, dataout=NA, x=formC(form_d, form_x,datause)$x[,-1], y=rlasso$resuse, xout=formC(form_d, form_x,dataout)$x[,-1], yout=rlasso$resout  , form_z, form_x,  nodesize=5, ntree=ntree, tune=tune)
    mz_x            <- rlasso$yhatout + forest$yhatout
    err.z           <- error(mz_x, dataout[,d])$err
    mis.z           <- error(mz_x, dataout[,d])$mis
    
    rz             <-  dataout[,d] -  mz_x
    
    
    rlasso          <- rlassoF(datause, dataout, form_xL, form_y, post, logit=FALSE)
    forest          <- RF(datause=NA, dataout=NA, x=formC(form_y, form_x,datause)$x[,-1], y=rlasso$resuse, xout=formC(form_y, form_x,dataout)$x[,-1], yout=rlasso$resout,  nodesize=5, ntree=ntree, tune=tune)
    ry              <- formC(form_y, form_x,dataout)$y - rlasso$yhatout  - forest$yhatout 
    err.y           <- error((rlasso$yhatout+forest$yhatout), dataout[,y])$err 
    
  }
  
  return(list(my_z1x=my_z1x, mz_x= mz_x, my_z0x=my_z0x, err.z = err.z,  err.yz0= err.yz0,  err.yz1=err.yz1, mis.z=mis.z, ry=ry , rz=rz, err.y=err.y));
  
}  




















