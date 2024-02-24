rm(list=ls(all=TRUE))

##Load pa
library(data.table)
library(Matrix)
library(foreach)
library(glmnet)
library(Hmisc) 
library(grid) 
library(lattice)
library(Formula)
library(ggplot2) 
library(rms)
library(survival)
library(foreign)
library(rmda)
library(DynNom)
library(pROC) 
library(ggplot2) 
library(epiDisplay)
library(questionr)
library(MASS)

a<-read.csv("path/train.csv",sep=",",header=TRUE) #read a table containing all deep learning + handcrafted features and lymph node status (non-metastatic=0 ; metastatic=1)
x1=as.matrix(a[,3:2594])  #column 3 to 2594 are the imaging features
y1=a[,2]    #column 2 is the ALN status

##use LASSO to select key features
fit <- glmnet(x1, y1, alpha=1,family = 'binomial')
set.seed(1)
fit_cv <- cv.glmnet(x1, y1, alpha=1, family = 'binomial',type.measure='mse')
plot(fit_cv)
get_coe <- function(the_fit,the_lamb){
  Coefficients <- coef(the_fit, s = the_lamb)
  Active.Index <- which(Coefficients != 0)
  Active.Coefficients <- Coefficients[Active.Index]
  re <- data.frame(rownames(Coefficients)[Active.Index],Active.Coefficients)
  re <- data.table('var_names'=rownames(Coefficients)[Active.Index],
                   'coef'=Active.Coefficients)
  re$expcoef <- exp(re$coef)
  write.csv(re, "path/feature_selected.csv") #save selected features
  return(re[order(expcoef)])
}
get_coe(fit_cv,fit_cv$lambda.min)

get_plot<- function(the_fit,the_fit_cv,the_lamb,toplot = seq(1,50,2)){
  Coefficients <- coef(the_fit, s = the_lamb)
  Active.Index <- which(Coefficients != 0)
  coeall <- coef(the_fit, s = the_fit_cv$lambda[toplot])
  coe <- coeall[Active.Index[-1],]
  ylims=c(-max(abs(coe)),max(abs(coe)))
  sp <- spline(log(the_fit_cv$lambda[toplot]),coe[1,],n=100)
  plot(sp,type='l',col =1,lty=1, 
       ylim = ylims,ylab = 'Coefficient', xlab = 'log(lambda)') 
  abline(h=0) 
  for(i in c(2:nrow(coe))){
    lines(spline(log(the_fit_cv$lambda[toplot]),coe[i,],n=1000),
          col =i,lty=1,lwd=2)
  }
  legend("bottomright",legend=rownames(coe),col=c(1:nrow(coe)),
         lty=c(1:nrow(coe)),
         cex=0.5)
}
get_plot(fit,fit_cv,exp(log(fit_cv$lambda.min)-1))

## compile the deep learning signature based on selected features
a$sig<-  
  a$LBP_38*-261.052888243926+
  a$LBP_172*1730.08075258608+
  a$LBP_198*3537.8776627219+
  a$LBP_206*-787.685264541501+
  a$f_135*-0.440843877143403+
  a$f_252*-0.208303096345564+
  a$f_272*-0.94454919695448+
  a$f_311*-2.12216970110515+
  a$f_374*-1.35003181264965+
  a$f_375*0.118704643234093+
  a$f_384*-0.115396233370636+
  a$f_400*-0.019389626360867+
  a$f_457*-0.119944463154932+
  a$f_542*0.440136732553323+
  a$f_558*0.503005269063104+
  a$f_644*0.452116911192224+
  a$f_700*-0.154679223443357+
  a$f_762*0.0974640095529039+
  a$f_779*0.213331325152889+
  a$f_782*-0.104276171814186+
  a$f_802*0.433883778245264+
  a$f_835*-0.712429789754987+
  a$f_883*0.165780249703388+
  a$f_895*0.0202823190484662+
  a$f_912*-0.409514216899134+
  a$f_1028*-0.312322792689046+
  a$f_1036*0.070728373937216+
  a$f_1037*1.25183579741937+
  a$f_1076*-1.15999862449485+
  a$f_1144*-0.0235035419916596+
  a$f_1145*-1.22181303157639+
  a$f_1170*0.0986493081227398+
  a$f_1222*0.76367357268216+
  a$f_1304*1.42323265885209+
  a$f_1313*1.03352503126835+
  a$f_1417*-2.1912110377454+
  a$f_1494*0.957396980893063+
  a$f_1506*-0.308703651476319+
  a$f_1523*-0.551035007385812+
  a$f_1556*-0.649483746812133+
  a$f_1557*-0.129048549627542+
  a$f_1606*-0.304032832989252+
  a$f_1612*0.828748397680478+
  a$f_1644*0.125743450884978+
  a$f_1802*0.296950561807491+
  a$f_1874*-0.125575034280155+
  a$f_1891*-0.154800722008559+
  a$f_1995*-0.222124087821026+
  a$f_2033*0.101169630011681
  
write.csv(a, "path/sig.csv") #save the signature

####After getting the deep learning signature, doctors can use the nomogram to calculate the possibility of lymph node metastasis

