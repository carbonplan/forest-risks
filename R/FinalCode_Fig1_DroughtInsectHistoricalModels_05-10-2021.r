#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------ Forest risk mapping models: Fig 1D-F code - 05/11/21 ----------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Author: William Anderegg (anderegg@utah.edu), University of Utah

library(rworldmap)
library(MASS)
library(SDMTools)
library(raster)
library(RNetCDF)
library(scales)
library(car)
library(RColorBrewer)
library(betareg)
library(pgirmess)
library(geosphere)
library(ape)


#---------------------------------- Pull in FIA mort data -------------------------------------
dir <- ""
fianew <- read.csv(paste(dir, "FIA-TerraClim-Wide-v17-04-14-2021.csv", sep = ""), header = T)
fialonga <- read.csv(paste(dir, "FIA-TerraClim-Long-1990.1999-v17-04-14-2021.csv", sep = ""), header = T)
fialongb <- read.csv(paste(dir, "FIA-TerraClim-Long-2000.2009-v17-04-14-2021.csv", sep = ""), header = T)
fialongc <- read.csv(paste(dir, "FIA-TerraClim-Long-2010.2019-v17-04-14-2021.csv", sep = ""), header = T)
FTpred_ins <- read.csv(paste(dir, "Insect_ForTypToPredict_04-21-2021.csv", sep = ""), header = T)
FTpred_drt <- read.csv(paste(dir, "Drought_ForTypToPredict_04-21-2021.csv", sep = ""), header = T)
# FTs that meet the minimum 20 mortality threshold and cross-validated AUC>0.6

# Processing projection and a useful logical function
proj2 <- CRS("+proj=longlat +datum=WGS84 +ellps=WGS84")
"%!in%" <- function(x, y) !("%in%"(x, y))



# Get functional trait data from Trugman et al. (2020) PNAS
p50a <- open.nc(paste(dir, "CWM_P50_025Deg.nc", sep = ""))
p50m <- var.get.nc(p50a, "CWM_P50")
dim(p50m)
p50r <- var.get.nc(p50a, "mean_range_within_site")
dim(p50r)
lat.tr <- var.get.nc(p50a, "lat")
dim(lat.tr)
lon.tr <- var.get.nc(p50a, "lon")
dim(lon.tr)

# Screen FIA data to remove fire, human disturbance, timber cutting, CONDPROP>0.3
fianew1 <- fianew[which(fianew[, 8] >= 0.3 & fianew[, 59] != "True" & fianew[, 63] != "True" & fianew[, 71] != "True" & fianew[, 18] > 1), ]

# Extract trait data for all FIA CONDs
traitdata <- array(dim = c(102453, 2))
for (i in 1:102453) {
  lati <- fianew1[i, 1]
  loni <- fianew1[i, 2]
  latcell <- which.min(abs(lati - lat.tr))
  loncell <- which.min(abs(loni - lon.tr))
  traitdata[i, 1] <- p50m[latcell, loncell]
  traitdata[i, 2] <- p50r[latcell, loncell]
}

traitlong <- array(dim = c(498410, 2))
for (i in 1:498410) {
  lati <- fialonga[i, 1]
  loni <- fialonga[i, 2]
  latcell <- which.min(abs(lati - lat.tr))
  loncell <- which.min(abs(loni - lon.tr))
  traitlong[i, 1] <- p50m[latcell, loncell]
  traitlong[i, 2] <- p50r[latcell, loncell]
}



#------------------------------------------- Processing FIA data --------------------------------------------
# For Obs, build a clean data-frame with dependent variable, predictor variables, FORTYP, lat/lon
fianew1[is.na(fianew1[, 23]) == "TRUE", 23] <- 0 # Impute 0s where no mort was measured
fianew1[is.na(fianew1[, 27]) == "TRUE", 27] <- 0 # Impute 0s where no mort was measured for insect mort
fianew1[which(fianew1[, 151] < -16), 151] <- -16 # Set lower PDSI bounds
fianew1[which(fianew1[, 151] > 16), 151] <- 16 # Set upper PDSI bounds
fianew1[which(fianew1[, 152] < -16), 152] <- -16 # Set lower PDSI bounds
fianew1[which(fianew1[, 152] > 16), 152] <- 16 # Set upper PDSI bounds
fianew1[which(fianew1[, 153] < -16), 153] <- -16 # Set lower PDSI bounds
fianew1[which(fianew1[, 153] > 16), 153] <- 16 # Set upper PDSI bounds

# Combine all cond data and variables into a dataframe with columns: MortDrt, MortIns, 18 clim vars, 2 age vars, 8 traits, lat,lon, FORTYP
fia.newm <- as.data.frame(cbind(
  (fianew1[, 23] / fianew1[, 18]) / (fianew1[, 15] - fianew1[, 14]), fianew1[, 27] * (fianew1[, 23] / fianew1[, 18]) / (fianew1[, 15] - fianew1[, 14]),
  scale(fianew1[, 127]), scale(fianew1[, 128]), scale(fianew1[, 129]), scale(fianew1[, 139]), scale(fianew1[, 140]), scale(fianew1[, 141]),
  scale(fianew1[, 151]), scale(fianew1[, 152]), scale(fianew1[, 153]), scale(fianew1[, 163]), scale(fianew1[, 164]), scale(fianew1[, 165]),
  scale(fianew1[, 175]), scale(fianew1[, 176]), scale(fianew1[, 177]), scale(fianew1[, 187]), scale(fianew1[, 188]), scale(fianew1[, 189]),
  scale(fianew1[, 3]^2), scale(fianew1[, 3]),
  scale(traitdata[, 1]), scale(traitdata[, 2]),
  (fianew1[, 19] - fianew1[, 18]), (fianew1[, 15] - fianew1[, 14]),
  fianew1[, 1], fianew1[, 2], fianew1[, 4], fianew1[, 18]
))

# Second, screen out NaN values that will mess up the model
fia2 <- fia.newm[which(fia.newm[, 1] != "NaN" & fia.newm[, 5] != "NaN" & fia.newm[, 3] != "NaN" & fia.newm[, 9] != "NaN" & fia.newm[, 29] != "NaN" & fia.newm[, 21] != "NaN" & fia.newm[, 23] != "NaN" & fia.newm[, 1] < 1.01 & is.na(fia.newm[, 2]) == "FALSE"), ]

# Third, scale up to 0.25x0.25 degree for model fitting
dummy <- raster(ncols = 244, nrows = 100, xmn = -126, xmx = -65, ymn = 25, ymx = 50, crs = proj2) # 0.25 degree dummy raster
fia.g1 <- array(dim = c(0, 30))
for (i in 1:112) {
  ftype <- unique(fia2[, 29])[i]
  d1 <- data.frame(fia2[which(fia2[, 29] == ftype), ])
  v1 <- rasterize(d1[, c(28, 27)], dummy, d1[, 1], fun = mean, background = NA, mask = FALSE, na.rm = T)
  v1i <- rasterize(d1[, c(28, 27)], dummy, d1[, 2], fun = mean, background = NA, mask = FALSE, na.rm = T)
  v1b <- rasterToPoints(v1)
  v1bi <- (as.matrix(as.vector(v1i)))[is.na(as.matrix(as.vector(v1i))) == "FALSE"]
  stack <- array(dim = c(length(v1bi), 0))
  for (j in 1:24) {
    varh <- rasterize(d1[, c(28, 27)], dummy, d1[, (j + 2)], fun = mean, background = NA, mask = FALSE, na.rm = T)
    var1 <- (as.matrix(as.vector(varh)))[is.na(as.matrix(as.vector(varh))) == "FALSE"]
    stack <- cbind(stack, var1)
  }
  ba <- rasterize(d1[, c(28, 27)], dummy, d1[, 30], fun = sum, background = NA, mask = FALSE, na.rm = T)
  ba0 <- (as.matrix(as.vector(ba)))[is.na(as.matrix(as.vector(ba))) == "FALSE"]
  fia.out <- cbind(v1b[, 3], v1bi, stack, v1b[, 1:2], rep(ftype, length(v1bi)), ba0)
  fia.g1 <- rbind(fia.g1, fia.out)
}



#------------------------------------------- Fit mortality functions --------------------------------------------------
# GLM control parameters - bump up maxit to help with convergence
gm.ctl <- glm.control(epsilon = 1e-8, maxit = 500, trace = FALSE)

mod_insect_proj <- function(fitdata, minthres, ftypcol, preddata) {
  colnames(preddata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13")
  colnames(fitdata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")
  mod1 <- array(dim = c(0, 7))
  for (i in 1:112) {
    ftype <- unique(fitdata[, ftypcol])[i]
    non_zero <- ifelse(fitdata[which(fitdata[, ftypcol] == ftype), 1] > 0, 1, 0)
    d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero) # Historical mortality data
    preddata1 <- data.frame(preddata[which(preddata[, 10] == ftype), ]) # Projection data
    # Set up the output matrix
    drt.out <- array(dim = c(dim(preddata1)[1], 7))
    drt.out[, 1] <- ftype # FORTYPCD
    drt.out[, 2] <- preddata1[, 12] # lat
    drt.out[, 3] <- preddata1[, 11] # lon
    drt.out[, 7] <- preddata1[, 13] # BAlive0 for the FORTYP

    # If FORTYP has non-meaningful historical model, set future projections == mean historical
    if (ftype %!in% FTpred_ins[, 1]) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Predicted mort = mean historical where Nmort<minthres
    if (ftype %!in% FTpred_ins[, 1]) mod1 <- rbind(mod1, drt.out)
    if (ftype %!in% FTpred_ins[, 1]) print(paste("FORTYP prediction skipped due inadequate historical model: ", ftype, sep = ""))
    if (ftype %!in% FTpred_ins[, 1]) next

    # If FORTYP has Nmort<minthres, set future projections == mean historical
    if (sum(d1[, 11], na.rm = T) < minthres) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Pred mort = mean hist where Nmort<minthres
    if (sum(d1[, 11], na.rm = T) < minthres) mod1 <- rbind(mod1, drt.out)
    if (sum(d1[, 11], na.rm = T) < minthres) print(paste("FORTYP prediction skipped due Nmort<minthres: ", ftype, sep = ""))
    if (sum(d1[, 11], na.rm = T) < minthres) next

    # For FORTYPs with adequate historical models
    if (sum(non_zero, na.rm = T) > 0) d1 <- d1[which(d1[, 1] < quantile(d1[, 1], 0.995)), ]
    if (dim(d1)[1] == 0) d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero)

    # Construct historical mortality model
    m1 <- glm(non_zero ~ V2 + V3 + V4 + V5 + V6 + V7 + V8, data = d1, family = binomial(link = logit), control = gm.ctl)
    m2 <- betareg(V1 ~ V2 + V3 + V4 + V5 + V6 + V7 + V8, data = subset(d1, non_zero == 1))

    # Predict with future climate projections
    pred1 <- predict(m1, newdata = preddata1[, 2:8], se = TRUE, type = "response")
    pred2 <- predict(m2, newdata = preddata1[, 2:8], se = TRUE, type = "response")
    pred1a <- ifelse(pred1$fit >= 0.5, 1, 0)
    pred3 <- pred2 * pred1a
    pred4 <- pred2 * pred1$fit
    # 		Store model output
    drt.out[, 5] <- pred3
    drt.out[, 6] <- pred4 # Predicted (expected value) mort frac
    mod1 <- rbind(mod1, drt.out)
  }
  print(paste("FORTYPs projected: ", length(unique(mod1[which(mod1[, 5] != "NA"), 1])), sep = ""))
  print(paste("FORTYPs completed: ", i, sep = ""))
  return(mod1)
}

insect_histmodel <- function(fitdata, minthres, ftypcol) {
  colnames(fitdata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13")
  mod1 <- array(dim = c(0, 7))
  for (i in 1:112) {
    ftype <- unique(fitdata[, ftypcol])[i]
    non_zero <- ifelse(fitdata[which(fitdata[, ftypcol] == ftype), 1] > 0, 1, 0)
    d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero)

    drt.out <- array(dim = c(dim(d1)[1], 7))
    drt.out[, 1] <- d1[, ftypcol] # FORTYPCD
    drt.out[, 2] <- d1[, 12] # lat
    drt.out[, 3] <- d1[, 11] # lon
    drt.out[, 4] <- d1[, 1] # Obs mort frac
    drt.out[, 7] <- d1[, 13] # BAlive0 for the FORTYP

    # If FORTYP has non-meaningful historical model, set future projections == mean historical
    if (ftype %!in% FTpred_ins[, 1]) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Predicted mort = mean historical where Nmort<minthres
    if (ftype %!in% FTpred_ins[, 1]) mod1 <- rbind(mod1, drt.out)
    if (ftype %!in% FTpred_ins[, 1]) print(paste("FORTYP prediction skipped due inadequate historical model: ", ftype, sep = ""))
    if (ftype %!in% FTpred_ins[, 1]) next

    # If FORTYP has Nmort<minthres, set future projections == mean historical
    if (sum(d1[, 14], na.rm = T) < minthres) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Predicted mort = mean historical where Nmort<minthres
    if (sum(d1[, 14], na.rm = T) < minthres) mod1 <- rbind(mod1, drt.out)
    if (sum(d1[, 14], na.rm = T) < minthres) print(paste("FORTYP prediction skipped due Nmort<minthres: ", ftype, sep = ""))
    if (sum(d1[, 14], na.rm = T) < minthres) next

    # For FORTYPs with adequate historical models, remove outliers
    if (sum(non_zero, na.rm = T) > 0) d1 <- d1[which(d1[, 1] < quantile(d1[, 1], 0.995)), ]
    if (dim(d1)[1] == 0) d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero)

    m1 <- glm(non_zero ~ V2 + V3 + V4 + V5 + V6 + V7 + V8, data = d1, family = binomial(link = logit), control = gm.ctl)
    m2 <- betareg(V1 ~ V2 + V3 + V4 + V5 + V6 + V7 + V8, data = subset(d1, non_zero == 1))

    pred1 <- predict(m1, newdata = d1[, 2:8], se = TRUE, type = "response")
    pred2 <- predict(m2, newdata = d1[, 2:8], se = TRUE, type = "response")
    pred1a <- ifelse(pred1$fit >= 0.5, 1, 0)
    pred3 <- pred1$fit
    pred4 <- pred2 * pred1$fit

    if (dim(drt.out)[1] != dim(d1)[1]) drt.out <- array(dim = c(dim(d1)[1], 7))
    drt.out[, 1] <- d1[, ftypcol] # FORTYPCD
    drt.out[, 2] <- d1[, 12] # lat
    drt.out[, 3] <- d1[, 11] # lon
    drt.out[, 4] <- d1[, 1] # Obs mort frac
    drt.out[, 7] <- d1[, 13] # BAlive0 for the FORTYP

    drt.out[, 5] <- pred3
    drt.out[, 6] <- pred4 # Predicted (expected value) mort frac
    mod1 <- rbind(mod1, drt.out)
  }
  print(paste("FORTYPs modeled: ", length(unique(mod1[which(mod1[, 5] != "NA"), 1])), sep = ""))
  print(paste("FORTYPs completed: ", i, sep = ""))
  return(mod1)
}






#----------------- FIAlong climate data preprocessing function
prepclim <- function(histclim, climdata, histcols, futcols) {
  climdata[which(climdata[, 6] < -16), 6] <- -16 # Set lower PDSI bounds
  climdata[which(climdata[, 6] > 16), 6] <- 16 # Set upper PDSI bounds
  climdata[which(climdata[, 13] < -16), 13] <- -16 # Set lower PDSI bounds
  climdata[which(climdata[, 13] > 16), 13] <- 16 # Set upper PDSI bounds
  histclim[which(histclim[, 25] < -16), 25] <- -16
  histclim[which(histclim[, 25] > 16), 25] <- 16
  histclim[which(histclim[, 26] < -16), 26] <- -16
  histclim[which(histclim[, 26] > 16), 26] <- 16
  print(histclim[1, histcols])
  print(climdata[1, futcols])
  # Get historical mean and SD for drought models
  hist1 <- array(dim = c(6, 2))
  for (i in 1:6) {
    hist1[i, 1] <- mean(histclim[, histcols[i]], na.rm = T)
    hist1[i, 2] <- sd(histclim[, histcols[i]], na.rm = T)
  }
  fia.futa <- as.data.frame(cbind(rep(1, 498410), (climdata[, futcols[1]] - hist1[1, 1]) / hist1[1, 2], (climdata[, futcols[2]] - hist1[2, 1]) / hist1[2, 2], (climdata[, futcols[3]] - hist1[3, 1]) / hist1[3, 2], (climdata[, futcols[4]] - hist1[4, 1]) / hist1[4, 2], (climdata[, futcols[5]] - hist1[5, 1]) / hist1[5, 2], (climdata[, futcols[6]] - hist1[6, 1]) / hist1[6, 2], scale(fialong[, 3]), scale(traitlong[, 2]), fialong[, 7], fialong[, 1], fialong[, 2], fialong[, 4]))
  finalclim <- fia.futa[which(fia.futa[, 1] != "NaN" & fia.futa[, 5] != "NaN" & fia.futa[, 3] != "NaN" & fia.futa[, 7] != "NaN" & fia.futa[, 10] != "NaN" & fia.futa[, 8] != "NaN" & is.na(fia.futa[, 2]) == "FALSE"), ]
  return(finalclim)
}

# Function when both inputs are FIAlong
prepclim1 <- function(histclim, climdata, histcols, futcols) {
  climdata[which(climdata[, 25] < -16), 25] <- -16 # Set lower PDSI bounds
  climdata[which(climdata[, 25] > 16), 25] <- 16 # Set upper PDSI bounds
  climdata[which(climdata[, 26] < -16), 26] <- -16 # Set lower PDSI bounds
  climdata[which(climdata[, 26] > 16), 26] <- 16 # Set upper PDSI bounds
  histclim[which(histclim[, 25] < -16), 25] <- -16
  histclim[which(histclim[, 25] > 16), 25] <- 16
  histclim[which(histclim[, 26] < -16), 26] <- -16
  histclim[which(histclim[, 26] > 16), 26] <- 16
  print(histclim[1, histcols])
  print(climdata[1, futcols])
  # Get historical mean and SD for drought models
  hist1 <- array(dim = c(6, 2))
  for (i in 1:6) {
    hist1[i, 1] <- mean(histclim[, histcols[i]], na.rm = T)
    hist1[i, 2] <- sd(histclim[, histcols[i]], na.rm = T)
  }
  fia.futa <- as.data.frame(cbind(rep(1, 498410), (climdata[, futcols[1]] - hist1[1, 1]) / hist1[1, 2], (climdata[, futcols[2]] - hist1[2, 1]) / hist1[2, 2], (climdata[, futcols[3]] - hist1[3, 1]) / hist1[3, 2], (climdata[, futcols[4]] - hist1[4, 1]) / hist1[4, 2], (climdata[, futcols[5]] - hist1[5, 1]) / hist1[5, 2], (climdata[, futcols[6]] - hist1[6, 1]) / hist1[6, 2], scale(fialonga[, 3]), scale(traitlong[, 2]), fialonga[, 7], fialonga[, 1], fialonga[, 2], fialonga[, 4]))
  finalclim <- fia.futa[which(fia.futa[, 1] != "NaN" & fia.futa[, 5] != "NaN" & fia.futa[, 3] != "NaN" & fia.futa[, 7] != "NaN" & fia.futa[, 10] != "NaN" & fia.futa[, 8] != "NaN" & is.na(fia.futa[, 2]) == "FALSE"), ]
  return(finalclim)
}




#-------------------- Function to rasterize and weight model mortality projections by observed/current BA
weight.proj1 <- function(raster1, grid, mortcol) {
  mort.bafi <- rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), 7], fun = mean, background = NA, mask = FALSE, na.rm = TRUE)
  mort.predfi <- rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), mortcol], fun = mean, background = NA, mask = FALSE, na.rm = TRUE)
  for (i in 2:length(unique(raster1[, 1]))) {
    mort.predfi <- addLayer(mort.predfi, rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), mortcol], fun = mean, background = NA, mask = FALSE, na.rm = TRUE))
    mort.bafi <- addLayer(mort.bafi, rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), 7], fun = mean, background = NA, mask = FALSE, na.rm = TRUE))
  }
  mort.predwi <- weighted.mean(mort.predfi, mort.bafi, na.rm = T)
  return(mort.predwi)
}




#-------------------------------- Run model and create figures
# Fig 1E
insect.hist <- insect_histmodel(fia.g1[, c(2, 5, 6, 11, 13, 15, 20, 22, 24, 29, 28, 27, 30)], 20, 10)

# Scale up by basal area
insect.obs <- weight.proj1(insect.hist, dummy, 4)

# writeRaster(insect.obs, paste(dir, "Fig1E_InsectModel_FIAwide-ObsMort_05-08-2021", sep=""),  format='GTiff')

# Fig 1F models
histcols6 <- c(19, 23, 25, 30, 32, 34)
insect.outa <- mod_insect_proj(fia.g1[, c(2, 5, 6, 11, 13, 15, 20, 22, 24, 29)], 20, 10, prepclim1(fialonga, fialonga, histcols6, histcols6))
insect.outb <- mod_insect_proj(fia.g1[, c(2, 5, 6, 11, 13, 15, 20, 22, 24, 29)], 20, 10, prepclim1(fialongb, fialongb, histcols6, histcols6))
insect.outc <- mod_insect_proj(fia.g1[, c(2, 5, 6, 11, 13, 15, 20, 22, 24, 29)], 20, 10, prepclim1(fialongc, fialongc, histcols6, histcols6))

# Make Fig 1F
insect1 <- insect.outa
insect1[, 6] <- rowMeans(cbind(insect.outa[, 6], insect.outb[, 6], insect.outc[, 6])) # Average modeled mortality across 3 baselines
mort.insects.modlong1 <- weight.proj1(insect1, dummy, 6)

# writeRaster(mort.insects.modlong1, paste(dir, "Fig1F_InsectModel_ModeledFIAlongEnsembleHistMort_05-08-2021", sep=""),  format='GTiff')




#-------------------------------------------- DROUGHT MODELS (Fig 1C/D)
mod_drought_proj <- function(fitdata, minthres, ftypcol, preddata) {
  colnames(preddata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13")
  colnames(fitdata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")
  mod1 <- array(dim = c(0, 7))
  for (i in 1:112) {
    ftype <- unique(fitdata[, ftypcol])[i]
    non_zero <- ifelse(fitdata[which(fitdata[, ftypcol] == ftype), 1] > 0, 1, 0)
    d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero) # Historical mortality data
    preddata1 <- data.frame(preddata[which(preddata[, 10] == ftype), ]) # Projection data

    # Set up the output matrix
    drt.out <- array(dim = c(dim(preddata1)[1], 7))
    drt.out[, 1] <- ftype # FORTYPCD
    drt.out[, 2] <- preddata1[, 12] # lat
    drt.out[, 3] <- preddata1[, 11] # lon
    drt.out[, 7] <- preddata1[, 13] # Biomass for the FORTYP

    # If FORTYP has non-meaningful historical model, set future projections == mean historical
    if (ftype %!in% FTpred_drt[, 1]) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Predicted mort = mean historical where Nmort<minthres
    if (ftype %!in% FTpred_drt[, 1]) mod1 <- rbind(mod1, drt.out)
    if (ftype %!in% FTpred_drt[, 1]) print(paste("FORTYP prediction skipped due inadequate historical model: ", ftype, sep = ""))
    if (ftype %!in% FTpred_drt[, 1]) next

    # If FORTYP has Nmort<minthres, set future projections == mean historical
    if (sum(d1[, 11], na.rm = T) < minthres) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Pred mort = mean hist where Nmort<minthres
    if (sum(d1[, 11], na.rm = T) < minthres) mod1 <- rbind(mod1, drt.out)
    if (sum(d1[, 11], na.rm = T) < minthres) print(paste("FORTYP prediction skipped due Nmort<minthres: ", ftype, sep = ""))
    if (sum(d1[, 11], na.rm = T) < minthres) next

    # For FORTYPs with adequate historical models, outlier screen
    if (sum(non_zero, na.rm = T) > 0) d1 <- d1[which(d1[, 1] < quantile(d1[, 1], 0.995)), ]
    if (dim(d1)[1] == 0) d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero)

    # Construct historical mortality model
    m1 <- glm(non_zero ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = d1, family = binomial(link = logit), control = gm.ctl)
    m2 <- betareg(V1 ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = subset(d1, non_zero == 1))

    # Predict with future climate projections
    pred1 <- predict(m1, newdata = preddata1[, 2:9], se = TRUE, type = "response")
    pred2 <- predict(m2, newdata = preddata1[, 2:9], se = TRUE, type = "response")
    pred1a <- ifelse(pred1$fit >= 0.5, 1, 0)
    pred3 <- pred2 * pred1a
    pred4 <- pred2 * pred1$fit

    # 		Store model output
    drt.out[, 5] <- pred3
    drt.out[, 6] <- pred4 # Predicted (expected value) mort frac
    mod1 <- rbind(mod1, drt.out)
  }
  print(paste("FORTYPs projected: ", length(unique(mod1[which(mod1[, 5] != "NA"), 1])), sep = ""))
  print(paste("FORTYPs completed: ", i, sep = ""))
  return(mod1)
}


drought_histmodel <- function(fitdata, minthres, ftypcol) {
  colnames(fitdata) <- c("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13")
  mod1 <- array(dim = c(0, 7))
  for (i in 1:112) {
    ftype <- unique(fitdata[, ftypcol])[i]
    non_zero <- ifelse(fitdata[which(fitdata[, ftypcol] == ftype), 1] > 0, 1, 0)
    d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero)

    drt.out <- array(dim = c(dim(d1)[1], 7))
    drt.out[, 1] <- d1[, ftypcol] # FORTYPCD
    drt.out[, 2] <- d1[, 12] # lat
    drt.out[, 3] <- d1[, 11] # lon
    drt.out[, 4] <- d1[, 1] # Obs mort frac
    drt.out[, 7] <- d1[, 13] # BAlive0 for the FORTYP

    # If FORTYP has non-meaningful historical model, set modeled values == mean historical
    if (ftype %!in% FTpred_drt[, 1]) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Predicted mort = mean historical where Nmort<minthres
    if (ftype %!in% FTpred_drt[, 1]) mod1 <- rbind(mod1, drt.out)
    if (ftype %!in% FTpred_drt[, 1]) print(paste("FORTYP prediction skipped due inadequate historical model: ", ftype, sep = ""))
    if (ftype %!in% FTpred_drt[, 1]) next

    # If FORTYP has Nmort<minthres, set modeled values == mean historical
    if (sum(d1[, 14], na.rm = T) < minthres) drt.out[, 6] <- mean(d1[, 1], na.rm = T) # Predicted mort = mean historical where Nmort<minthres
    if (sum(d1[, 14], na.rm = T) < minthres) mod1 <- rbind(mod1, drt.out)
    if (sum(d1[, 14], na.rm = T) < minthres) print(paste("FORTYP prediction skipped due Nmort<minthres: ", ftype, sep = ""))
    if (sum(d1[, 14], na.rm = T) < minthres) next

    # For FORTYPs with adequate historical models, remove outliers
    if (sum(non_zero, na.rm = T) > 0) d1 <- d1[which(d1[, 1] < quantile(d1[, 1], 0.995)), ]
    if (dim(d1)[1] == 0) d1 <- data.frame(fitdata[which(fitdata[, ftypcol] == ftype), ], non_zero)

    # Build models
    m1 <- glm(non_zero ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = d1, family = binomial(link = logit), control = gm.ctl)
    m2 <- betareg(V1 ~ V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = subset(d1, non_zero == 1))

    pred1 <- predict(m1, newdata = d1[, 2:9], se = TRUE, type = "response")
    pred2 <- predict(m2, newdata = d1[, 2:9], se = TRUE, type = "response")
    pred1a <- ifelse(pred1$fit >= 0.5, 1, 0)
    pred3 <- pred1$fit
    pred4 <- pred2 * pred1$fit

    if (dim(drt.out)[1] != dim(d1)[1]) drt.out <- array(dim = c(dim(d1)[1], 7))
    drt.out[, 1] <- d1[, ftypcol] # FORTYPCD
    drt.out[, 2] <- d1[, 12] # lat
    drt.out[, 3] <- d1[, 11] # lon
    drt.out[, 4] <- d1[, 1] # Obs mort frac
    drt.out[, 7] <- d1[, 13] # BAlive0 for the FORTYP

    drt.out[, 5] <- pred3
    drt.out[, 6] <- pred4 # Predicted (expected value) mort frac
    mod1 <- rbind(mod1, drt.out)
  }
  print(paste("FORTYPs modeled: ", length(unique(mod1[which(mod1[, 5] != "NA"), 1])), sep = ""))
  print(paste("FORTYPs completed: ", i, sep = ""))
  return(mod1)
}



#-------------------------------- Run model and create figures
# Fig 1C
drought.hist <- drought_histmodel(fia.g1[, c(1, 5, 7, 11, 13, 17, 18, 22, 24, 29, 28, 27, 30)], 20, 10)

# Scale up by basal area
drought.obs <- weight.proj1(drought.hist, dummy, 4)

# writeRaster(drought.obs, paste(dir, "Fig1C_DroughtModel_FIAwide-ObsMort_05-08-2021", sep=""),  format='GTiff')

# Fig 1D
histcols1a <- c(19, 24, 25, 30, 31, 35)
drought.outa <- mod_drought_proj(fia.g1[, c(1, 5, 7, 11, 13, 17, 18, 22, 24, 29)], 20, 10, prepclim1(fialonga, fialonga, histcols1a, histcols1a))
drought.outb <- mod_drought_proj(fia.g1[, c(1, 5, 7, 11, 13, 17, 18, 22, 24, 29)], 20, 10, prepclim1(fialongb, fialongb, histcols1a, histcols1a))
drought.outc <- mod_drought_proj(fia.g1[, c(1, 5, 7, 11, 13, 17, 18, 22, 24, 29)], 20, 10, prepclim1(fialongc, fialongc, histcols1a, histcols1a))

# Make Fig 1D
drought1 <- drought.outa
drought1[, 6] <- rowMeans(cbind(drought.outa[, 6], drought.outb[, 6], drought.outc[, 6])) # Average modeled mortality across 3 baselines
mort.drought.modlong1 <- weight.proj1(drought1, dummy, 6)

# writeRaster(mort.drought.modlong1, paste(dir, "Fig1D_DroughtModel_ModeledFIAlongEnsembleHistMort_05-08-2021", sep=""),  format='GTiff')





#-------------------------------------- First-order reversal estimates
# Historical reversals for drought and insects in FIA data
drt.rev <- rep(0, 102453)
drt.rev[which(fianew1[, 27] < 0.01 & fianew1[, 19] < fianew1[, 18] & fianew1[, 31] < 0.01 & fianew1[, 35] < 0.01 & fianew1[, 39] < 0.01 & fianew1[, 47] < 0.01 & fianew1[, 55] == "False" & fianew1[, 59] == "False" & fianew1[, 63] == "False" & fianew1[, 71] == "False" & fianew1[, 75] == "False" & fianew1[, 79] == "False" & fianew1[, 83] == "False" & fianew1[, 151] < -2 & fianew1[, 67] == "False" & ((fianew1[, 19] - fianew1[, 18]) / -fianew1[, 18]) > 0.1)] <- 1

ins.rev <- rep(0, 102453)
ins.rev[which(fianew1[, 27] > 0.5 & fianew1[, 19] < fianew1[, 18] & ((fianew1[, 19] - fianew1[, 18]) / -fianew1[, 18]) > 0.1)] <- 1

# Calculated annualized drought and insect p(rev)
revhist1 <- array(dim = c(102453, 2))
for (i in 1:102453) {
  ftrev <- fianew1[i, 4]
  revhist1[i, 1] <- drt.rev[i] / mean(fianew1[which(fianew1[, 4] == ftrev), 15] - fianew1[which(fianew1[, 4] == ftrev), 14], na.rm = T)
  revhist1[i, 2] <- ins.rev[i] / mean(fianew1[which(fianew1[, 4] == ftrev), 15] - fianew1[which(fianew1[, 4] == ftrev), 14], na.rm = T)
}

# Calculate 100-year integrated risk %s of constant historical reversal annual probabilities
100 * dbinom(1, 100, mean(revhist1[, 1], na.rm = T)) # Drought
100 * dbinom(1, 100, mean(revhist1[, 2], na.rm = T)) # Insects


# Simulate 100-year integrated risk %s with Fig 2 multi-model mean CMIP6 projections
out <- array(dim = c(1000, 0))
perc1 <- c(1, 1.022, 1.063, 1.148, 1.151, 1.230, 1.380, 1.384, 1.799, 1.846) # Percent changes in BAmortfrac/yr from drought models compared to 1990-2010 baselines
for (i in 1:10) {
  dec1 <- rbinom(1000, 10, 0.002457 * perc1[i]) # Draw 1000 10-year draws with observed probability
  out <- cbind(out, dec1)
  sumout <- rowSums(out)
  sumout1 <- sum(ifelse(sumout > 0, 1, 0))
}
100 * sumout1 / 1000

# Insects
out <- array(dim = c(1000, 0))
perc2 <- c(1, 1.049, 1.042, 1.193, 1.0231, 1.136, 1.289, 1.197, 1.622, 1.701) # Percent changes in BAmortfrac/yr from insect models compared to 1990-2010 baselines
for (i in 1:10) {
  dec1 <- rbinom(1000, 10, 0.00250982 * perc2[i]) # Draw 1000 10-year draws with observed probability
  out <- cbind(out, dec1)
  sumout <- rowSums(out)
  sumout1 <- sum(ifelse(sumout > 0, 1, 0))
}
100 * sumout1 / 1000

# Fire
out <- array(dim = c(1000, 0))
hist1 <- read.csv(paste(dir, "historical_fire.csv", sep = ""), header = T)
histave <- mean(hist1[17:35, 2])
fut1 <- read.csv(paste(dir, "future_fire.csv", sep = ""), header = T)
perc3 <- c(rep(1, 19), (fut1[50:130, 3] - histave) / histave + 1)
for (i in 1:100) {
  dec1 <- rbinom(1000, 1, histave * perc3[i]) # Draw 1000 1-year draws with observed probability
  out <- cbind(out, dec1)
  sumout <- rowSums(out)
  sumout1 <- sum(ifelse(sumout > 0, 1, 0))
}
100 * sumout1 / 1000
