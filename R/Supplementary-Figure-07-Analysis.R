#--------------------------------------------------------------------------------------------------------------------------------

#------------------------- No-analog climate analysis of drought FIA risk models - 2/11/22 ----------------------

#--------------------------------------------------------------------------------------------------------------------------------
# Contact: William Anderegg, University of Utah (anderegg@utah.edu)

library(rworldmap)
library(MASS)
library(SDMTools)
library(raster)
library(RNetCDF)
library(scales)
library(car)
library(RColorBrewer)
library(nlme)
library(betareg)
library(pgirmess)
library(geosphere)
library(ape)


#---------------------------------- Pull in FIA mort data -------------------------------------
dir="/Users/billanderegg/Desktop/FRC-RiskCalculations/FIA_3-21/"
fianew <- read.csv(paste(dir,"FIA-TerraClim-Wide-v17-04-14-2021.csv", sep=""), header=T)
fialonga <- read.csv(paste(dir,"FIA-TerraClim-Long-1990.1999-v17-04-14-2021.csv", sep=""), header=T)
fialongb <- read.csv(paste(dir,"FIA-TerraClim-Long-2000.2009-v17-04-14-2021.csv", sep=""), header=T)
fialongc <- read.csv(paste(dir,"FIA-TerraClim-Long-2010.2019-v17-04-14-2021.csv", sep=""), header=T)
FTpred_drt <- read.csv(paste(dir,"Drought_ForTypToPredict_04-21-2021.csv", sep=""), header=T)
FTpred_drt <- FTpred_drt[1:61,2:5]
# CSV above is FORTYPs (N=53) with meaningful (cross-validated AUC>0.6) historical models to be used in future climate projections

# Processing projection and a useful logical function
proj2 <-  CRS("+proj=longlat +datum=WGS84 +ellps=WGS84")
'%!in%' <- function(x,y)!('%in%'(x,y))



# Get functional trait data (not used in insect projections)
# dir="/Users/BillsComputer/Desktop/Projects-Literature/ForestsClimatePolicy/RiskData/FIA_drought/"
p50a <- open.nc(paste(dir,"CWM_P50_025Deg.nc", sep=""))
p50m = var.get.nc(p50a,"CWM_P50")
dim(p50m)
p50r = var.get.nc(p50a,"mean_range_within_site")
dim(p50r)
lat.tr = var.get.nc(p50a,"lat")
dim(lat.tr)
lon.tr = var.get.nc(p50a,"lon")
dim(lon.tr)

fianew1 <- fianew[which(fianew[,8]>=0.3 & fianew[,59]!="True" & fianew[,63]!="True" & fianew[,71]!="True" & fianew[,18]>1),]
traitdata <- array(dim=c(102453,2))
for (i in 1: 102453){
	lati <- fianew1[i,1]
	loni <- fianew1[i,2]
	latcell <- which.min(abs(lati-lat.tr))
	loncell <- which.min(abs(loni-lon.tr))
	traitdata[i,1] <- p50m[latcell,loncell]
	traitdata[i,2] <- p50r[latcell,loncell]
}

traitlong <- array(dim=c(498410,2))
for (i in 1:498410){
	lati <- fialonga[i,1]
	loni <- fialonga[i,2]
	latcell <- which.min(abs(lati-lat.tr))
	loncell <- which.min(abs(loni-lon.tr))
	traitlong[i,1] <- p50m[latcell,loncell]
	traitlong[i,2] <- p50r[latcell,loncell]
}



#------------------------------------------- Processing/screening --------------------------------------------
# Subset by >0.3 condprop, no fire, cutting, human disturbance, BAlive0>1
fianew1 <- fianew[which(fianew[,8]>=0.3 & fianew[,59]!="True" & fianew[,63]!="True" & fianew[,71]!="True" & fianew[,18]>1),]

# For Obs, build a clean data-frame with dependent variable, predictor variables, FORTYP, lat/lon
fianew1[1,c(23,18,15,14,127,140,151,164,176,188,3,4,1,2)]
fianew1[is.na(fianew1[,23])=="TRUE",23] <- 0		# Impute 0s where no mort was measured
fianew1[is.na(fianew1[,27])=="TRUE",27] <- 0		# Impute 0s where no mort was measured for insect mort
fianew1[which(fianew1[,151]< -16),151] <- -16		# Set lower PDSI bounds
fianew1[which(fianew1[,151]> 16),151] <- 16			# Set upper PDSI bounds
fianew1[which(fianew1[,152]< -16),152] <- -16		# Set lower PDSI bounds
fianew1[which(fianew1[,152]> 16),152] <- 16			# Set upper PDSI bounds
fianew1[which(fianew1[,153]< -16),153] <- -16		# Set lower PDSI bounds
fianew1[which(fianew1[,153]> 16),153] <- 16			# Set upper PDSI bounds

# Combine all cond data and variables into a dataframe w/33 cols: MortDrt, MortIns, 18 clim vars, 2 age vars, 8 traits, lat,lon, FORTYP
fianew1[1,c(23,18,15,14,127,140,151,164,176,188,3,4,1,2)]
fianew1[1,c(127:129,139:141,151:153,163:165,175:177,187:189)]
fia.newm <- as.data.frame(cbind((fianew1[,23]/fianew1[,18])/(fianew1[,15]-fianew1[,14]), fianew1[,27]*(fianew1[,23]/fianew1[,18])/(fianew1[,15]-fianew1[,14]),
	scale(fianew1[,127]), scale(fianew1[,128]), scale(fianew1[,129]), scale(fianew1[,139]), scale(fianew1[,140]), scale(fianew1[,141]),
	scale(fianew1[,151]), scale(fianew1[,152]), scale(fianew1[,153]), scale(fianew1[,163]), scale(fianew1[,164]), scale(fianew1[,165]),
	scale(fianew1[,175]), scale(fianew1[,176]), scale(fianew1[,177]), scale(fianew1[,187]), scale(fianew1[,188]), scale(fianew1[,189]),
	scale(fianew1[,3]^2), scale(fianew1[,3]),
	scale(traitdata[,1]), scale(traitdata[,2]),
	(fianew1[,19]-fianew1[,18]), (fianew1[,15]-fianew1[,14]),
	fianew1[,1], fianew1[,2], fianew1[,4], fianew1[,18])
)

# Second, screen out NaN values that will mess up the model
fia2 <- fia.newm[which(fia.newm[,1]!="NaN" & fia.newm[,5]!="NaN" & fia.newm[,3]!="NaN" & fia.newm[,9]!="NaN" & fia.newm[,29]!="NaN" & fia.newm[,21]!="NaN" & fia.newm[,23]!="NaN" & fia.newm[,1]<1.01 & is.na(fia.newm[,2])=="FALSE"),]

# Third, scale up to 0.25x0.25 degree for model fitting
dummy <- raster(ncols=244, nrows=100, xmn=-126, xmx=-65, ymn=25, ymx=50, crs=proj2) #0.25 degree dummy raster
fia.g1 <- array(dim=c(0,30))
for (i in 1:112){
		ftype <- unique(fia2[,29])[i]
		d1 <- data.frame(fia2[which(fia2[,29]==ftype),])
		v1 <- rasterize(d1[,c(28,27)], dummy, d1[,1], fun=mean, background=NA, mask=FALSE, na.rm=T)
		v1i <- rasterize(d1[,c(28,27)], dummy, d1[,2], fun=mean, background=NA, mask=FALSE, na.rm=T)
		v1b <- rasterToPoints(v1)
		v1bi <- (as.matrix(as.vector(v1i)))[is.na(as.matrix(as.vector(v1i)))=="FALSE"]
		stack <- array(dim=c(length(v1bi),0))
		for (j in 1:24){
			varh <- rasterize(d1[,c(28,27)], dummy, d1[,(j+2)], fun=mean, background=NA, mask=FALSE, na.rm=T)
			var1 <- (as.matrix(as.vector(varh)))[is.na(as.matrix(as.vector(varh)))=="FALSE"]
			stack <- cbind(stack, var1)
		}
		ba <- rasterize(d1[,c(28,27)], dummy, d1[,30], fun=sum, background=NA, mask=FALSE, na.rm=T)
		ba0 <- (as.matrix(as.vector(ba)))[is.na(as.matrix(as.vector(ba)))=="FALSE"]
		fia.out <- cbind(v1b[,3], v1bi, stack, v1b[,1:2], rep(ftype, length(v1bi)), ba0)
		fia.g1 <- rbind(fia.g1, fia.out)
}

#----------------- FIAlong climate data preprocessing function
prepclim3 <- function(histclim, climdata, histcols, futcols){
	# climdata[which(climdata[,6]< -16),6] <- -16		# Set lower PDSI bounds
	# climdata[which(climdata[,6]> 16),6] <- 16			# Set upper PDSI bounds
	# climdata[which(climdata[,13]< -16),13] <- -16		# Set lower PDSI bounds
	# climdata[which(climdata[,13]> 16),13] <- 16			# Set upper PDSI bounds
	histclim[which(histclim[,25]< -16),25] <- -16
	histclim[which(histclim[,25]> 16),25] <- 16
	histclim[which(histclim[,26]< -16),26] <- -16
	histclim[which(histclim[,26]> 16),26] <- 16
	print(histclim[1,histcols])
	print(climdata[1,futcols])
	# Get historical mean and SD for drought models
	hist1 <- array(dim=c(6,2))
	for (i in 1:6){
		hist1[i,1] <- mean(histclim[,histcols[i]], na.rm=T)
		hist1[i,2] <- sd(histclim[,histcols[i]], na.rm=T)
	}
	fia.futa <- as.data.frame(cbind(rep(1,498410), (climdata[,futcols[1]]-hist1[1,1])/hist1[1,2], (climdata[,futcols[2]]-hist1[2,1])/hist1[2,2], (climdata[,futcols[3]]-hist1[3,1])/hist1[3,2],(climdata[,futcols[4]]-hist1[4,1])/hist1[4,2], (climdata[,futcols[5]]-hist1[5,1])/hist1[5,2], (climdata[,futcols[6]]-hist1[6,1])/hist1[6,2],  scale(fialonga[,3]), scale(traitlong[,2]), fialonga[,7], fialonga[,1], fialonga[,2], fialonga[,4]))
	finalclim <- fia.futa[which(fia.futa[,1]!="NaN" & fia.futa[,5]!="NaN" & fia.futa[,3]!="NaN" & fia.futa[,7]!="NaN" & fia.futa[,10]!="NaN" & fia.futa[,8]!="NaN" & is.na(fia.futa[,2])=="FALSE"),]
	return(finalclim)
}

#--------------------------- Climate analog extraction function
climanalog <- function(fitdata, preddata, minthres){
	colnames(preddata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13")
	colnames(fitdata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")
	# Set up the output matrix: 112 FT rows, columns of % no-analog for each of 6 climate variables
	drt.out <- array(dim=c(112, 7))
	for (i in 1:112){
		ftype <- unique(fitdata[, 10])[i]
		non_zero <- ifelse(fitdata[which(fitdata[, 10]==ftype),1] > 0, 1, 0)
		d1 <- data.frame(fitdata[which(fitdata[, 10]==ftype),], non_zero)			# Historical mortality/climate data
		preddata1 <- data.frame(preddata[which(preddata[,10]==ftype),])					# Projection data
		drt.out[i,1] <- ftype															# FORTYPCD

		# If FORTYP has non-meaningful historical model, set future projections == mean historical
		if(ftype %!in% FTpred_drt[,1]) drt.out[i,2:7]<-0		# Predicted mort = mean historical where Nmort<minthres ==> 0% no-analog climate
		if(ftype %!in% FTpred_drt[,1]) print(paste("FORTYP prediction skipped due inadequate historical model: ", ftype, sep=""))
		if(ftype %!in% FTpred_drt[,1]) next

		# If FORTYP has Nmort<minthres, set future projections == mean historical
		if(sum(d1[,11], na.rm=T)<minthres) drt.out[i,2:7]<-0		# Pred mort = mean hist where Nmort<minthres ==> 0% no-analog climate
		if(sum(d1[,11], na.rm=T)<minthres) print(paste("FORTYP prediction skipped due Nmort<minthres: ", ftype, sep=""))
		if(sum(d1[,11], na.rm=T)<minthres) next

		# For FORTYPs with adequate historical models
		no_an <- array(dim=c(6, 1))
		for (j in 1:6){
			Nout <- rep(0,dim(preddata1)[1])
			range1 <- range(d1[,1+j], na.rm=T)
			Nout[(preddata1[,1+j]<range1[1] | preddata1[,1+j]>range1[2])]<-1
			no_an[j,1] <- 100*sum(Nout, na.rm=T)/dim(preddata1)[1]
		}
		drt.out[i,2:7]<-no_an
	}
	print(paste("FORTYPs projected: ", length(unique(drt.out[which(drt.out[,5]!=0),1])), sep=""))
	print(paste("FORTYPs completed: ", i, sep=""))
	return(drt.out)
}


# Now test it on a single model
climatedata <- read.csv(paste(dir,"climdata/", models2[1], '.',  scenario[9], '.', member[1], "-", datest[9], ".", dateend[9], '-v18-05-03-2021.csv', sep=""))[,2:22]
#testout <- climanalog(fia.g1[, c(1,5,7,11,13,17,18,22,24,29)], prepclim3(fialonga,climatedata,histcols1,futcols1), 20)
testout <- climanalog(fia.newm[, c(1,5,7,11,13,17,18,22,24,29)], prepclim3(fialonga,climatedata,histcols1,futcols1), 20)




#--------------------------- Process all climate models
# Read in model list
#dir="/Users/BillsComputer/Desktop/Projects-Literature/ForestsClimatePolicy/RiskData/FIA_drought/FIA-CMIP6-v12-03-12-21/"
models1 <- read.csv(paste(dir,"models.csv", sep=""), header=T)

# Correct columns to use for historical and future climate CSVs
histcols1 <- c(19,24,25,30,31,35)
futcols1 <- c(11,19,13,21,15,9)

# SSP 370
# Set up model and date matrices to loop over
models2 <- unique(models1[,2])
datest <- c(2010,2020,2030,2040,2050,2060,2070,2080,2090)
dateend <- c(2019,2029,2039,2049,2059,2069,2079,2089,2099)
scenario <- c(rep("ssp370", 9))
member <-  models1[c(1,5,9,13,17,21),4]

# Loop over models and decades
noantemp <- array(dim=c(112,1))
noanpcp <- array(dim=c(112,1))
noanpsdi <- array(dim=c(112,1))
noancwd <- array(dim=c(112,1))
noanpet <- array(dim=c(112,1))
noanvpd <- array(dim=c(112,1))
for (i in 1:6){
	for (j in 8:9){
		# Read in future climate projection
		#climatedata <- read.csv(paste('https://carbonplan.blob.core.windows.net/carbonplan-scratch/forests/quantile-mapping/FIA-CMIP6-Long-', models2[i], '.',  scenario[j], '.', member[i], "-", datest[j], ".", dateend[j], '-v14-03-17-2021.csv', sep=""))
		climatedata <- read.csv(paste(dir,"climdata/", models2[i], '.',  scenario[j], '.', member[i], "-", datest[j], ".", dateend[j], '-v18-05-03-2021.csv', sep=""))[,2:22]

		# Do mortality projection
		noan1 <- climanalog(fia.newm[, c(1,5,7,11,13,17,18,22,24,29)], prepclim3(fialonga,climatedata,histcols1,futcols1), 20)
		noantemp <- cbind(noantemp,noan1[,3])
		noanpcp <- cbind(noanpcp,noan1[,2])
		noanpsdi <- cbind(noanpsdi,noan1[,4])
		noancwd <- cbind(noancwd,noan1[,5])
		noanpet <- cbind(noanpet,noan1[,6])
		noanvpd <- cbind(noanvpd,noan1[,7])
	}
}
print(Sys.time())

tempall <- rowMeans(noantemp[,2:13])
pcpall <- rowMeans(noanpcp[,2:13])
pdsiall <- rowMeans(noanpsdi[,2:13])
cwdall <- rowMeans(noancwd[,2:13])
petall <- rowMeans(noanpet[,2:13])
vpdall <- rowMeans(noanvpd[,2:13])

# Output data for figure
write.csv(cbind(tempall,pcpall,pdsiall,cwdall,petall,vpdall), paste(dir,"noanalog_ssp370_4-5-22.csv", sep=""))

# Total no-analog calculation
length(c(tempall[which(tempall>40)], pcpall[which(pcpall>40)], pdsiall[which(pdsiall>40)], cwdall[which(cwdall>40)], petall[which(petall>40)], vpdall[which(vpdall>40)]))		# 116 combinations
100*116/(6*112)					# 17% of all end-of-century SSP370 projections
