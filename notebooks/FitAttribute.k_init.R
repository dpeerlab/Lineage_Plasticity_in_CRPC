FitAttribute=function(Network,AttributeMat,Prob, k, kmeansinit = NULL){
	#October 30,2017
	#This is the code used to fit the attributed SBM 
	#Inputs:

		#Network: NxN adjacency matrix (can be sparse)
		#AttributeMat: NxP matrix of attributes
			#This can be a probaibility of being in 1 of P Classes
		#k: The number of communities
		#A binary indicator indicating whether attributes is probability
			#0: no these are not probabilities
			#1: yes these are probabilities

#Please make sure you have igraph
library('igraph')

#################
#Initialization: Initialize partition of nodes into communities 
#################
N=nrow(AttributeMat)

if (is.null(kmeansinit)) {
	if(Prob==1){
		kmeansinit=apply(AttributeMat,1,function(x) which(x==max(x))[1])
	} else{
		MakeNet=graph.adjacency(Network,mode='undirected', weighted=T)

		#Use network (louvain) to initialize
		kmeansinit=membership(cluster_spinglass(MakeNet, spins=k))

		#use network (sbm to initialize)
		 # MixerRes<-getModel(mixer(as.matrix(Network),qmin=4,qmax=9))
		 # 	Taus<-t(MixerRes$Taus)
		 # 	kmeansinit<-apply(Taus,1,function(x) which(x==max(x))[1])
		#print(kmeansinit)

		#kmeansinit=kmeans(Network,centers=4)$cluster
		# ##part to plot the baseline
		# jBrew=brewer.pal(n=6,name='Dark2')
		# ColorAssn=jBrew[kmeansinit]
		# quartz()
		# V(MakeGraph)$color=ColorAssn
		# plot(MakeGraph,vertex.size=4,vertex.label=NA,layout=l,edge.width=E(MakeGraph)$weight*.2,main='Clustering just with network')
		# stop('test complete')
		# #use attributes to initialize
		# #kmeansinit=kmeans(Network,centers=7)$cluster
	}	
}


#print(kmeansinit)
#create

#note change this if you want to input a k
k=max(kmeansinit)


#create the feature mean and covariance measures
StoreMean<-list()
StoreCov<-list()

##compute means and covariance matrices for each community###
for(kk in 1:k){
	Inds<-which(kmeansinit==kk)
	if(length(Inds)==0){
		StoreMean[[kk]]=matrix(0,ncol=ncol(AttributeMat),nrow=1)
		StoreCov[[k]]=matrix(0,nrow=ncol(AttributeMat),ncol=ncol(AttributeMat))
	}
	else{
	StoreMean[[kk]]<-matrix(colSums(matrix(AttributeMat[Inds,],ncol=ncol(AttributeMat)))/length(Inds),ncol=ncol(AttributeMat))
	StoreCov[[kk]]<-EstCovMat(matrix(AttributeMat[Inds,],ncol=ncol(AttributeMat)))
}
	if(sum(StoreCov[[kk]])==0){
		StoreCov[[kk]]<-StoreCov[[kk]]+0.000001
	}
}

	
##compute the initial error##
print('this is the initial error!!!')
ErrorVal=Error(AttributeMat,kmeansinit,StoreMean)
print(ErrorVal)

#compute an initial SBM connectivity profile 
ProbMat=matrix(0.01,nrow=N,ncol=k)

for(i in 1:length(kmeansinit)){
	ProbMat[i,kmeansinit[i]]=.99
}

Alpha=UpdateBlockParameters(ProbMat,Network)

#create pivector or the probability of being in each of the class (k length vector)
PiMat=apply(ProbMat,2,function(x) sum(x)/length(x))
logPi=log(PiMat)

#the first expectation is just the probability matrix
Expect=ProbMat
InitialObjective<-ObjectiveFunction(AttributeMat,StoreMean,StoreCov,Alpha,logPi,Network,Expect)
print(InitialObjective)
##########################
#begin iterative process
##########################
proceed=1
Diff=2
Objective=InitialObjective
objectivevec=c()
while(proceed>0 & Diff>1){
  	
  #update Mu and Cov
	for(m in 1:k){
		StoreMean[[m]]<-matrix(MuUpdate(Expect,AttributeMat,m))
		StoreCov[[m]]<-CovMatUpdate(StoreMean[[m]],Expect,AttributeMat,m)
	}
  #compute new expectations based on these updated parameters 
  ExpectNew<-ComputeExpectation(AttributeMat,StoreMean,StoreCov,Alpha,k,logPi,Expect,Network)
  
  #update the pi matrices of belonging to communities
 	PiMat<-colSums(ExpectNew)/nrow(ExpectNew)
	zeroInds<-which(PiMat==0)
	PiMat[zeroInds]<-0.000001
	logPi<-log(PiMat)

  #update SBM probability matrices
  AlphaNew<-UpdateBlockParameters(ExpectNew,Network)
  
  #update the new objective value
  NewObjective<-ObjectiveFunction(AttributeMat,StoreMean,StoreCov,AlphaNew,logPi,Network,ExpectNew)
  
  if(NewObjective>Objective){
  		proceed=1
  		##update everything##
  		Alpha=AlphaNew
  		Expect<-ExpectNew
  		Diff=abs(Objective-NewObjective)
  		Objective=NewObjective
      objectivevec<-c(objectivevec,NewObjective)
  }
  
  else{proceed=0
  		
  }
  print(objectivevec)

} #while
CommAssn=apply(Expect,1,function(x) which(x==max(x))[1])
print(CommAssn)

########

# jBrew=brewer.pal(n=6,name='Dark2')
# ColorAssn=jBrew[CommAssn]
# quartz()
# V(MakeGraph)$color=ColorAssn
# plot(MakeGraph,vertex.size=4,vertex.label=NA,layout=l,edge.width=E(MakeGraph)$weight*.2,main='Clustering just with attribute model')

#create and Output List: entry 1 is community assignment
#entry 2 is SBM matrix
#entry 3 is Mean Mat
#entry 4 is CoVMat

OutList=list()
OutList[[1]]=CommAssn
OutList[[2]]=Alpha
OutList[[3]]=StoreMean
OutList[[4]]=StoreCov

names(OutList)=c('Comm','SBMProb','Mean','Cov')

##now compute the new mean
print('this is the new error!!!')
ErrorVal=Error(AttributeMat,CommAssn,StoreMean)
print(ErrorVal)

OutList[['FinalError']] = ErrorVal
OutList
} #function end


#################################
######Helper Functions###########
#################################
UpdateBlockParameters<-function(ExpMat,Network){
	#This function updates block model parameters under the SBM model
	#Inputs: ExpMat: the n x k matrix of expectations
		#Network: adjacency matrix

		NewBlockM<-matrix(0,nrow=ncol(ExpMat),ncol=ncol(ExpMat))

		for(q in 1:ncol(NewBlockM)){
			for(l in 1:ncol(NewBlockM)){
				if(q<=l){
					denom<-0
					num<-0
					for(i in 1:nrow(Network)){
						for(j in 1:nrow(Network)){
							if(i<j){
								num<-num+(ExpMat[i,q]*ExpMat[j,l]*Network[i,j])
								denom<-denom+max(0.00000001,(ExpMat[i,q]*ExpMat[j,l]))
								
							}
						}
					}
					NewBlockM[q,l]<-num/denom
					NewBlockM[l,q]<-NewBlockM[q,l]
				}
			}
		}
		
NewBlockM

}

ObjectiveFunction<-function(AttributeMat,StoreMean,StoreCov,Alpha,piMat,Network,ExpMat){
	#computes objective based on current exp mat, and mv gaussian parameters
	#Inputs:
		#AttributeMat: the n x p matrix of attributes
		#StoreMean: the k length list of means
		#StoreCov: the k length list of covariances
		#Alpha: the kxk propensity matrix
		#piMat: the vector of log probabilities of cluster assignments
		#Network: adjacency matrix
		#ExpMat: the current matrix of expectationd

	ObjectiveFunction<-0

	#First compute attribute LL
	AttributeLL<-matrix(0,nrow=nrow(AttributeMat),ncol=length(piMat))
	for(j in 1:ncol(AttributeLL)){
		AttributeLL[,j]<-GaussianLikelihood(AttributeMat,StoreMean[[j]],StoreCov[[j]])
	}
	zeroInds=which(AttributeLL==0)
	AttributeLL[zeroInds]<-0.0000001
	#AttributeLL<-log(AttributeLL)
	
	#Now compute graph LL
	GraphLL<-matrix(0,nrow=nrow(AttributeMat),ncol=length(piMat))
  for(c in 1:ncol(AttributeLL)){
  	#thing<-logproba(Network,ExpMat,Alpha,c)
  	thing<-logprobaSBM(Network,ExpMat,Alpha,c)
  	GraphLL[,c]<-thing
  }
  
  for(i in 1:nrow(AttributeMat)){
		for(kc in 1:length(piMat)){
		  ObjectiveFunction<-ObjectiveFunction+(ExpMat[i,kc]*AttributeLL[i,kc])+(ExpMat[i,kc]*GraphLL[i,kc])+(ExpMat[i,kc]*piMat[kc])
		}
	}
  ObjectiveFunction
}


GaussianLikelihood<-function(DM,mean,cov){
	library('mvtnorm')
	#This function is meant to compute p(x|z_i,theta), where theta is set of parameters
	#Inputs:
		#DM: nxp array of data
		#mean: n-length vector mean for cluster of interest
		#cov: pxp covariance matrix
	
	storeprob<-rep(0,nrow(DM))
	
	for(i in 1:nrow(DM)){
		mean=as.matrix(mean,nrow=1)
		
		draw<-dmvnorm(DM[i,],mean=mean,sigma=cov,log=TRUE)
		
		if(is.infinite(draw)){
			storeprob[i]<-0.00000001
		}
		else{storeprob[i]<-dmvnorm(DM[i,],mean=mean,sigma=cov,log=TRUE)}
	}
	
	storeprob

}

logprobaSBM<-function(DM,ExpMat,alphaMat,commind){
	#calculates p(a_i | z_ic)
	#inputs:
		#DM is nxn adjacency matrix
		#Exp mat is nxk matrix of expectations
		#alphamat is kxk matrix of probability parameters 
		#commind is community index of interest
	#output is n-length array of probabilities
	
	CalcNeighbors<-Neighbors(DM)
	#print(CalcNeighbors)
	#place to store log probabilities
	logprobvec<-rep(0,nrow(DM))

	#soft community assignments
	SoftCommAssn<-apply(ExpMat,1,function(x) which(x==max(x))[1])


for(i in 1:nrow(DM)){
		SpecificNeighbors<-CalcNeighbors[[i]]
		NonNeighbors<-setdiff(c(1:nrow(DM))[-i],SpecificNeighbors)
		logprobNeighbor<-0
		logprobNonNeighbor<-0
		for(c in 1:ncol(ExpMat)){
		if(length(SpecificNeighbors)==0){
			logprobNeighbor<-0
		}
		else{
			
		for(j in 1:length(SpecificNeighbors)){
			logprobNeighbor<-logprobNeighbor+ExpMat[SpecificNeighbors[j],c]*log(alphaMat[commind,c])
		}
		}
		#do same process for non neighbor
		
		for(k in 1:length(NonNeighbors)){
			logprobNonNeighbor<-logprobNonNeighbor+ExpMat[NonNeighbors[k],c]*log(1-alphaMat[commind,c])
		}
		
	} ##c
	FinalLogProb<-logprobNeighbor+logprobNonNeighbor
	logprobvec[i]<-FinalLogProb
	} ###for i in 1:nrow(DM)
	logprobvec
}

Neighbors<-function(AdjMat){
  #for each node determine the set of neighbors
  SetsOfNeighbors<-list()
  for(i in 1:nrow(AdjMat)){
  	Inds<-which(AdjMat[i,]!=0)
  	SetsOfNeighbors[[i]]<-Inds
  }
  SetsOfNeighbors
}

logsum<-function(vec){
	m=max(vec)
	s=log(sum(exp(vec-m)))+m
	s
}

ComputeExpectation<-function(AttributeMat,StoreMu,StoreCov,Alpha,k,logPi,TauMat,Network){
	#this finction takes means, covariances, etc and computes probability of node to community assn

#inputs
	#AttributeMat is the nxp matrix of attrobites
	#StoreMu is the list object of Mus
	#StoreCov is the list object of estimated covariances
	#Alpha is the propensity 
	#k is the number of expected communities
	#logPi is probability of being in each of the communities 
	#TauMat is the prior version of Tau

AttributeLL<-matrix(0,nrow=nrow(AttributeMat),ncol=k)
for(j in 1:ncol(AttributeLL)){
		AttributeLL[,j]<-GaussianLikelihood(AttributeMat,StoreMu[[j]],StoreCov[[j]])
	}
	zeroInds<-which(AttributeLL==0)
	AttributeLL[zeroInds]<-0.00000001
	#AttributeLL<-log(AttributeLL)

###Now do the analog for graph #####
GraphLL<-matrix(0,nrow=nrow(AttributeMat),ncol=k)
for(c in 1:ncol(AttributeLL)){
	GraphLL[,c]<-logprobaSBM(Network,TauMat,Alpha,c)+logPi[c]
}

PreLogSum<-AttributeLL+GraphLL
LogSum<-apply(PreLogSum,1,function(x) logsum(x))
TauInit<-exp(PreLogSum-LogSum)
ZeroInds<-which(TauInit==0)
TauInit[ZeroInds]<-1*exp(-10)
TauInit
}

EstCovMat<-function(MVSamples){
	#computes estimated covariance matrix
	#mvsamples is nxp data matrix

	Means<-matrix(colSums(MVSamples)/nrow(MVSamples),ncol=1)
	EstCov<-matrix(0,nrow=ncol(MVSamples),ncol=ncol(MVSamples))
	for(i in 1:nrow(MVSamples)){
		RelData<-matrix(MVSamples[i,],ncol=1)

		Diff<-matrix(RelData-Means)
		EstCov<-EstCov+(Diff%*%t(Diff))
	}
	EstCov<-EstCov/nrow(MVSamples)
	# if(sum(EstCov==0)){
	# 	EstCov=EstCov+0.0000001
	# }

	EstCov
}

MuUpdate<-function(ExpMat,DataMat,commind){
	#this function performs update of mu for mixture of gaussian
	#assumes we have n observations and p dimensions, #k clusters
	#ExpMat: nxk matrix of probabilities 
		#ExpMat{ik} gives probability node i is in comm k
	#DataMat: Nxp matrix of data observations
	#commind: the cluster we are computing this for
	NewMu<-rep(0,ncol(DataMat))
	Bottom<-0
	for(i in 1:nrow(DataMat)){
		NewMu<-NewMu+(ExpMat[i,commind]*DataMat[i,])
		Bottom<-Bottom+ExpMat[i,commind]
	}
	NewMu<-NewMu/Bottom
	NewMu
}

CovMatUpdate<-function(Mu,ExpMat,DataMat,commind){
	#updates covariance matrix for mog
	#Assumes n observeations, p features, k clusters
	#Inputs:
		#Mu: the p-length vector of means
		#ExpMat: nxk matrix of expectations
		#DataMat: nxp matrix of data
		#commind: the community we are interested in updating 

	CovMat<-matrix(0,nrow=ncol(DataMat),ncol=ncol(DataMat))
	Bottom<-0
	#make sure Mu is a column vector
	Mu<-as.matrix(Mu,ncol=1)
	for(i in 1:nrow(DataMat)){
		#Extract row of DataMat and turn into column vector
		RelFeat<-as.matrix(DataMat[i,],ncol=1)
		DiffVec<-(RelFeat-Mu)
		CovMat<-CovMat+(ExpMat[i,commind]*(DiffVec%*%t(DiffVec)))
		Bottom<-Bottom+ExpMat[i,commind]
	}
	CovMat<-CovMat/Bottom
	
	CovMat
}

###compute the error function: what the baseline mean from prob dist over the nodes was 
Error=function(AttributeArray,Labels,MeanList){
	#We will subtract the true attributes (NxK matrix) from those inferred. 
	#Inputs
		#AttributeArray: NxP array of attributes
		#Labels: the N-length vecor of node labels
		#MeanList: the list of means

	#create a matrix to store the inferred means
	InferMean=matrix(0,nrow=nrow(AttributeArray),ncol=ncol(AttributeArray))
	for(i in 1:nrow(InferMean)){
		InferMean[i,]=MeanList[[Labels[i]]]
	}
DiffMat=AttributeArray-InferMean
NormVal=norm(DiffMat,type='F')
NormVal
}