#-----------------------------------------------------------------------------------------------------------------------------

#------------------- Forest risk mapping - Fig 2-3 (insect future projection) code - 03/29/21 --------------------------------

#-----------------------------------------------------------------------------------------------------------------------------
# Author: William Anderegg (anderegg@utah.edu)

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
dir <- ""
fianew <- read.csv(paste(dir, "FIA-TerraClim-Wide-v12-03-12-2021.csv", sep = ""), header = T)
fialong <- read.csv(paste(dir, "FIA-TerraClim-Long-v12-03-12-2021.csv", sep = ""), header = T)
FTpred_ins <- read.csv(paste(dir, "Insect_ForTypToPredict_03-22-2021.csv", sep = ""), header = T)
# CSV above is FORTYPs with meaningful (cross-validated AUC>0.6) historical models to be used in future climate projections

# Processing projection and a useful logical function
proj2 <- CRS("+proj=longlat +datum=WGS84 +ellps=WGS84")
"%!in%" <- function(x, y) !("%in%"(x, y))



# Get functional trait data
dir <- ""
p50a <- open.nc(paste(dir, "CWM_P50_025Deg.nc", sep = ""))
p50m <- var.get.nc(p50a, "CWM_P50")
dim(p50m)
p50r <- var.get.nc(p50a, "mean_range_within_site")
dim(p50r)
lat.tr <- var.get.nc(p50a, "lat")
dim(lat.tr)
lon.tr <- var.get.nc(p50a, "lon")
dim(lon.tr)

fianew1 <- fianew[which(fianew[, 8] >= 0.3 & fianew[, 59] != "True" & fianew[, 63] != "True" & fianew[, 71] != "True" & fianew[, 18] > 1), ]
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
  lati <- fialong[i, 1]
  loni <- fialong[i, 2]
  latcell <- which.min(abs(lati - lat.tr))
  loncell <- which.min(abs(loni - lon.tr))
  traitlong[i, 1] <- p50m[latcell, loncell]
  traitlong[i, 2] <- p50r[latcell, loncell]
}



#------------------------------------------- Processing/screening --------------------------------------------
# Subset by >0.3 condprop, no fire, cutting, human disturbance, BAlive0>1
fianew1 <- fianew[which(fianew[, 8] >= 0.3 & fianew[, 59] != "True" & fianew[, 63] != "True" & fianew[, 71] != "True" & fianew[, 18] > 1), ]

# For Obs, build a clean data-frame with dependent variable, predictor variables, FORTYP, lat/lon
fianew1[1, c(23, 18, 15, 14, 127, 140, 151, 164, 176, 188, 3, 4, 1, 2)]
fianew1[is.na(fianew1[, 23]) == "TRUE", 23] <- 0 # Impute 0s where no mort was measured
fianew1[is.na(fianew1[, 27]) == "TRUE", 27] <- 0 # Impute 0s where no mort was measured for insect mort
fianew1[which(fianew1[, 151] < -16), 151] <- -16 # Set lower PDSI bounds
fianew1[which(fianew1[, 151] > 16), 151] <- 16 # Set upper PDSI bounds
fianew1[which(fianew1[, 152] < -16), 152] <- -16 # Set lower PDSI bounds
fianew1[which(fianew1[, 152] > 16), 152] <- 16 # Set upper PDSI bounds
fianew1[which(fianew1[, 153] < -16), 153] <- -16 # Set lower PDSI bounds
fianew1[which(fianew1[, 153] > 16), 153] <- 16 # Set upper PDSI bounds

# Combine all cond data and variables into a dataframe w/33 cols: MortDrt, MortIns, 18 clim vars, 2 age vars, 8 traits, lat,lon, FORTYP
fianew1[1, c(23, 18, 15, 14, 127, 140, 151, 164, 176, 188, 3, 4, 1, 2)]
fianew1[1, c(127:129, 139:141, 151:153, 163:165, 175:177, 187:189)]
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
emptygrid <- raster(ncols = 244, nrows = 100, xmn = -126, xmx = -65, ymn = 25, ymx = 50, crs = proj2) # 0.25 degree emptygrid raster
fia.g1 <- array(dim = c(0, 30))
for (i in 1:112) {
  ftype <- unique(fia2[, 29])[i]
  d1 <- data.frame(fia2[which(fia2[, 29] == ftype), ])
  v1 <- rasterize(d1[, c(28, 27)], emptygrid, d1[, 1], fun = mean, background = NA, mask = FALSE, na.rm = T)
  v1i <- rasterize(d1[, c(28, 27)], emptygrid, d1[, 2], fun = mean, background = NA, mask = FALSE, na.rm = T)
  v1b <- rasterToPoints(v1)
  v1bi <- (as.matrix(as.vector(v1i)))[is.na(as.matrix(as.vector(v1i))) == "FALSE"]
  stack <- array(dim = c(length(v1bi), 0))
  for (j in 1:24) {
    varh <- rasterize(d1[, c(28, 27)], emptygrid, d1[, (j + 2)], fun = mean, background = NA, mask = FALSE, na.rm = T)
    var1 <- (as.matrix(as.vector(varh)))[is.na(as.matrix(as.vector(varh))) == "FALSE"]
    stack <- cbind(stack, var1)
  }
  ba <- rasterize(d1[, c(28, 27)], emptygrid, d1[, 30], fun = sum, background = NA, mask = FALSE, na.rm = T)
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
    drt.out[, 7] <- preddata1[, 13] # Summed BAlive0 for the FORTYP

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
    drt.out[, 5] <- pred3 # Fitted mort frac
    drt.out[, 6] <- pred4 # Predicted (expected value) mort frac
    mod1 <- rbind(mod1, drt.out)
  }
  print(paste("FORTYPs projected: ", length(unique(mod1[which(mod1[, 5] != "NA"), 1])), sep = ""))
  print(paste("FORTYPs completed: ", i, sep = ""))
  return(mod1)
}




#----------------- FIAlong climate data preprocessing function
prepclim <- function(climdata, histcols, futcols) {
  climdata[which(climdata[, 6] < -16), 6] <- -16 # Set lower PDSI bounds
  climdata[which(climdata[, 6] > 16), 6] <- 16 # Set upper PDSI bounds
  climdata[which(climdata[, 12] < -16), 12] <- -16 # Set lower PDSI bounds
  climdata[which(climdata[, 12] > 16), 12] <- 16 # Set upper PDSI bounds
  print(fialong[1, histcols])
  print(climdata[1, futcols])
  # Get historical mean and SD for drought models
  hist1 <- array(dim = c(6, 2))
  for (i in 1:6) {
    hist1[i, 1] <- mean(fialong[, histcols[i]], na.rm = T)
    hist1[i, 2] <- sd(fialong[, histcols[i]], na.rm = T)
  }
  fia.futa <- as.data.frame(cbind(rep(1, 498410), (climdata[, futcols[1]] - hist1[1, 1]) / hist1[1, 2], (climdata[, futcols[2]] - hist1[2, 1]) / hist1[2, 2], (climdata[, futcols[3]] - hist1[3, 1]) / hist1[3, 2], (climdata[, futcols[4]] - hist1[4, 1]) / hist1[4, 2], (climdata[, futcols[5]] - hist1[5, 1]) / hist1[5, 2], (climdata[, futcols[6]] - hist1[6, 1]) / hist1[6, 2], scale(fialong[, 3]), scale(traitlong[, 2]), fialong[, 7], fialong[, 1], fialong[, 2], fialong[, 4]))
  finalclim <- fia.futa[which(fia.futa[, 1] != "NaN" & fia.futa[, 5] != "NaN" & fia.futa[, 3] != "NaN" & fia.futa[, 7] != "NaN" & fia.futa[, 10] != "NaN" & fia.futa[, 8] != "NaN" & is.na(fia.futa[, 2]) == "FALSE"), ]
  return(finalclim)
}


#----------------- Upscaling function
# Function to rasterize and weight model mortality projections by observed/current BA
weight.proj <- function(raster1, grid, mortcol) {
  mort.bafi <- rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), 7], fun = sum, background = NA, mask = FALSE, na.rm = TRUE)
  mort.predfi <- rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[1]), mortcol], fun = mean, background = NA, mask = FALSE, na.rm = TRUE)
  for (i in 2:length(unique(raster1[, 1]))) {
    mort.predfi <- addLayer(mort.predfi, rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), mortcol], fun = mean, background = NA, mask = FALSE, na.rm = TRUE))
    mort.bafi <- addLayer(mort.bafi, rasterize(raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), c(2, 3)], grid, raster1[which(raster1[, 1] == unique(raster1[, 1])[i]), 7], fun = sum, background = NA, mask = FALSE, na.rm = TRUE))
  }
  mort.predwi <- weighted.mean(mort.predfi, mort.bafi, na.rm = T)
  return(mort.predwi)
}



#--------------------------------------- Make projections (Fig 2)
# Read in model list
models1 <- read.csv(paste(dir, "models.csv", sep = ""), header = T)

# Correct columns to use for historical and future climate CSVs
histcols <- c(19, 22, 26, 30, 32, 36)
futcols <- c(10, 11, 6, 19, 8, 21)


#-------------- SSP 585
# Set up model and date matrices to loop over
models2 <- unique(models1[, 2])
datest <- c(2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090)
dateend <- c(2009, 2019, 2029, 2039, 2049, 2059, 2069, 2079, 2089, 2099)
scenario <- c(rep("historical", 1), rep("ssp585", 9))
member <- models1[c(1, 5, 9, 13, 17, 21), 4]

# Loop over models and decades (1 decade historical plus 9 decades SSP)
for (i in 1:6) {
  for (j in 1:10) {
    # Read in future climate projection
    climatedata <- read.csv(paste("https://carbonplan.blob.core.windows.net/carbonplan-scratch/forests/quantile-mapping/FIA-CMIP6-Long-", models2[i], ".", scenario[j], ".", member[i], "-", datest[j], ".", dateend[j], "-v14-03-17-2021.csv", sep = ""))

    # Do mortality projection
    mortproj <- mod_insect_proj(fia.g1[, c(2, 5, 8, 9, 13, 15, 19, 22, 24, 29)], 21, 10, prepclim(climatedata, histcols, futcols))

    # Scale up to 0.25 degree & Write raster output
    mort.insects.proj <- weight.proj(mortproj, emptygrid, 6)
    writeRaster(mort.insects.proj, paste(dir, "insectmap1/", "InsectModelProjection_", models2[i], ".", scenario[j], ".", member[i], "-", datest[j], ".", dateend[j], "-v14climate", "_3-22-2021", sep = ""), format = "GTiff")
  }
}




#-------------- SSP 370
# Set up model and date matrices to loop over
models2 <- unique(models1[, 2])
datest <- c(2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090)
dateend <- c(2019, 2029, 2039, 2049, 2059, 2069, 2079, 2089, 2099)
scenario <- c(rep("ssp370", 9))
member <- models1[c(1, 5, 9, 13, 17, 21), 4]

# Loop over models and decades
for (i in 1:6) {
  for (j in 1:9) {
    # Read in future climate projection
    climatedata <- read.csv(paste("https://carbonplan.blob.core.windows.net/carbonplan-scratch/forests/quantile-mapping/FIA-CMIP6-Long-", models2[i], ".", scenario[j], ".", member[i], "-", datest[j], ".", dateend[j], "-v14-03-17-2021.csv", sep = ""))

    # Do mortality projection
    mortproj <- mod_insect_proj(fia.g1[, c(2, 5, 8, 9, 13, 15, 19, 22, 24, 29)], 21, 10, prepclim(climatedata, histcols, futcols))

    # Scale up to 0.25 degree & Write raster output
    mort.insects.proj <- weight.proj(mortproj, emptygrid, 6)
    writeRaster(mort.insects.proj, paste(dir, "insectmap1/", "InsectModelProjection_", models2[i], ".", scenario[j], ".", member[i], "-", datest[j], ".", dateend[j], "-v14climate", "_3-22-2021", sep = ""), format = "GTiff")
  }
}




#-------------- SSP 245
# Set up model and date matrices to loop over
models2 <- unique(models1[, 2])
datest <- c(2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090)
dateend <- c(2019, 2029, 2039, 2049, 2059, 2069, 2079, 2089, 2099)
scenario <- c(rep("ssp245", 9))
member <- models1[c(1, 5, 9, 13, 17, 21), 4]

# Loop over models and decades
for (i in 1:6) {
  for (j in 1:9) {
    # Read in future climate projection
    climatedata <- read.csv(paste("https://carbonplan.blob.core.windows.net/carbonplan-scratch/forests/quantile-mapping/FIA-CMIP6-Long-", models2[i], ".", scenario[j], ".", member[i], "-", datest[j], ".", dateend[j], "-v14-03-17-2021.csv", sep = ""))

    # Do mortality projection
    mortproj <- mod_insect_proj(fia.g1[, c(2, 5, 8, 9, 13, 15, 19, 22, 24, 29)], 21, 10, prepclim(climatedata, histcols, futcols))

    # Scale up to 0.25 degree & Write raster output
    mort.insects.proj <- weight.proj(mortproj, emptygrid, 6)
    writeRaster(mort.insects.proj, paste(dir, "insectmap1/", "InsectModelProjection_", models2[i], ".", scenario[j], ".", member[i], "-", datest[j], ".", dateend[j], "-v14climate", "_3-22-2021", sep = ""), format = "GTiff")
  }
}
