library(RLT)

args <- commandArgs(trailingOnly=TRUE)
train_path  <- args[1]
test_path   <- args[2]
output_path <- args[3]
task        <- args[4]

# Charger les données
train <- read.csv(train_path)
test  <- read.csv(test_path)

y <- train[,1]
X <- train[,-1]

Xtest <- test

# Paramètres similaires à ceux de l'article
n <- nrow(X)
nmin <- floor(n^(1/3))

# Paramètres pour muting (variable muting)
param.m <- list(
  muting = TRUE,
  muting.minimum = 0.5   # 50% minimum muting
)

# Paramètres pour linear combination split
param.lc <- list(
  linear.combination = TRUE,
  combsplit = TRUE,
  combsplit.level = 2    # équivalent à comb_size = 2
)

# Appel RLT
fit <- RLT(
  x = X,
  y = y,
  ntrees = 100,
  nmin = nmin,
  task = task,
  param.m = param.m,
  param.lc = param.lc
)

# Prédictions
pred <- predict_rlt(fit, Xtest)

write.csv(pred, output_path, row.names=FALSE)
