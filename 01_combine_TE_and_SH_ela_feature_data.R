wd = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wd)
library(doParallel)
library(tidyverse)

obj_quantile = 0.01
budget_factor = 100


aas_files = list.files('data/alg', full.names = T)
meta_data = read_csv('data/meta_data.csv')
bench_info = read_csv('data/benchmark_info.csv')
######################################################################################
cl = makeCluster(detectCores())
registerDoParallel(cl)
benchmark_files = foreach(i = 1:nrow(bench_info), .combine = rbind) %dopar% {
  library(dplyr)
  library(readr)
  library(stringr)
  library(foreach)
  bench = bench_info[[i, 1]]
  ins = bench_info[[i, 2]]
  meta = filter(meta_data, bname == bench)
  tmp_files = aas_files[str_detect(aas_files, paste(bench, ins, '', sep = "_"))]
  avail = ifelse(length(tmp_files) == 4, T, F)
  
  data = lapply(tmp_files, read_csv) %>%
    bind_rows()
  
  budget = ifelse(nrow(data) == meta$dim*budget_factor*4*20, T, F)
  
  
  data.frame(bname = bench, instance = ins, avail = avail, budget = budget)
  
  
}
stopCluster(cl)
rm(cl)
registerDoSEQ()


######################################################################################
# Read algorithm data and get distribution of targets
cl = makeCluster(detectCores())
registerDoParallel(cl)
targets = foreach(i = 1:nrow(bench_info), .combine = rbind) %dopar% {
  library(dplyr)
  library(readr)
  library(stringr)
  bench = bench_info[[i, 1]]
  ins = bench_info[[i, 2]]
  meta = filter(meta_data, bname == bench)
  avail = filter(benchmark_files, bname == bench, instance == ins)
  if (avail$avail) {
    tmp_files = aas_files[str_detect(aas_files, paste(bench, ins, '', sep = "_"))]
    
    data = lapply(tmp_files, read_csv) %>%
      bind_rows() %>%
      filter(nfev <= (100 * meta$dim))
    
    # Get dynamic target
    target = quantile(data$obj_val, obj_quantile)
    
    data.frame(bench = bench, instance = ins, target = target)
  }
}
stopCluster(cl)
rm(cl)
registerDoSEQ()

write.csv(targets, "data/targets.csv", row.names = F)



######################################################################################
# Read algorithm data and filter down to 100*D budget
cl = makeCluster(detectCores())
registerDoParallel(cl)
aas_results = foreach(i = 1:nrow(bench_info), .combine = rbind) %dopar% {
  library(dplyr)
  library(readr)
  library(stringr)
  library(foreach)
  bench = bench_info[[i, 1]]
  ins = bench_info[[i, 2]]
  meta = filter(meta_data, bname == bench)
  avail = filter(benchmark_files, bname == bench, instance == ins)
  if (avail$avail) {
    tmp_files = aas_files[str_detect(aas_files, paste(bench, ins, '', sep = "_"))]
    
    data = lapply(tmp_files, read_csv) %>%
      bind_rows() %>%
      filter(nfev <= (100 * meta$dim))
    
    # Get dynamic target
    target = quantile(data$obj_val, obj_quantile)
    
    # Identify the first target hit
    data = data %>%
      filter(obj_val <= target) %>%
      arrange(solver, rep, nfev) %>%
      group_by(solver, rep) %>%
      filter(row_number() == 1) %>%
      ungroup()
    
    
    ggplot(data, aes(obj_val, fill = solver)) +
      geom_boxplot()
    # Calculate ERT
    ert_data = foreach(solv = c("RS", "EA", "OP", "SM"), .combine = rbind) %do% {
      tmp = data %>%
        filter(solver == solv)
      
      if(nrow(tmp) > 0) {
        penalty = 20 - nrow(tmp)
        ert = (sum(tmp$nfev) + (penalty * budget_factor * meta$dim))/20
      } else {
        ert = 10 * 20 * budget_factor * meta$dim
      }
      
      data.frame(bench = bench, solver = solv, instance = ins, ert = ert)
    }
    
    ert_data
  }
}
stopCluster(cl)
rm(cl)
registerDoSEQ()

write.csv(aas_results, "data/raw_ert.csv", row.names = F)
aas_results = read_csv("data/raw_ert.csv")

######################################################################################

# Get VBS performance
cl = makeCluster(detectCores())
registerDoParallel(cl)
vbs = foreach(i = 1:nrow(bench_info), .combine = rbind) %dopar% {
  library(dplyr)
  library(readr)
  library(stringr)
  library(foreach)
  bname = bench_info[[i, 1]]
  ins = bench_info[[i, 2]]
  
  tmp = aas_results %>%
    filter(bench == bname, instance == ins) %>%
    arrange(ert)
  
  ert = tmp$ert[1]
  solver = tmp$solver[1]
  
  data.frame(bench = bname, instance = ins, solver = solver, ert = ert)
}
stopCluster(cl)
rm(cl)
registerDoSEQ()

# Check for NA's (should not occur!)
vbs[is.na(vbs$ert), ]
#vbs = drop_na(vbs)


write.csv(vbs, 'data/label_data.csv', row.names = F)
######################################################################################

# Calculate relative ERT values
cl = makeCluster(detectCores())
registerDoParallel(cl)
rel_ert = foreach(i = 1:nrow(bench_info), .combine = rbind) %dopar% {
  library(dplyr)
  library(readr)
  library(stringr)
  library(foreach)
  bname = bench_info[[i, 1]]
  ins = bench_info[[i, 2]]
  
  tmp = aas_results %>%
    filter(bench == bname, instance == ins) %>%
    arrange(ert)
  
  ert = tmp$ert[1]
  
  tmp$ert = tmp$ert/ert
  tmp
}
stopCluster(cl)
rm(cl)
registerDoSEQ()

write.csv(rel_ert, 'data/rel_ert.csv', row.names = F)



######################################################################################

vbs %>%
  group_by(solver) %>%
  summarize(count = n()) %>%
  ungroup()

mean(vbs$ert)
######################################################################################
# Identify SBS
sbs = aas_results %>%
  group_by(solver) %>%
  summarize(ert = mean(ert)) %>%
  ungroup()

sbs = sbs %>%
  arrange(ert) %>%
  filter(row_number() == 1)

# relERT performance of SBS
sbs_rel_ert = rel_ert %>%
  group_by(solver) %>%
  summarize(rel_ert = mean(ert)) %>%
  ungroup() %>%
  arrange(rel_ert)

sbs_rel_ert = sbs_rel_ert %>%
  arrange(rel_ert) %>%
  filter(row_number() == 1)


######################################################################################
# Calculate ELA

ela_files = list.files('data/ela', full.names = T)

cl = makeCluster(detectCores())
registerDoParallel(cl)
ela_results = foreach(i = 1:nrow(bench_info), .combine = rbind) %dopar% {
  library(dplyr)
  library(readr)
  library(stringr)
  library(foreach)
  bench = bench_info[[i, 1]]
  ins = bench_info[[i, 2]]
  meta = filter(meta_data, bname == bench)
  tmp_files = ela_files[str_detect(ela_files, paste(bench, ins, '', sep = "_"))]
  
  ela = foreach(x = tmp_files, .combine = rbind) %do% {
    current_files = read_csv(x)
    # auskommentiert für Oversampling. andernfalls wieder rien
    #type = substr(x, 10, 11)
    type = 'SH'
    
    current_files = current_files %>%
      mutate(type = type)
    
    current_files
    
  }
  ela
  
}
stopCluster(cl)
rm(cl)
registerDoSEQ()

write.csv(ela_results, 'data/ela_data_sh.csv', row.names = F)

######### To delete:
ggplot(ela_results, aes(model_mae)) +
  geom_histogram() +
  facet_wrap(bench ~ .) +
  theme_light()

###### do all into one rbind
ela_sh = read_csv('data/ela_data_sh.csv')
model_mae = ela_sh$model_mae
ela_sh = ela_sh %>%
  select(-model_mae)
ela_sh$model_mae = model_mae


ela_normal = read_csv('data/ela_data.csv')%>%
  filter(type == 'TE')
ela_normal$model_mae = NA


ela_all = rbind(ela_sh, ela_normal)
write.csv(ela_all, 'data/ela_data_all.csv', row.names = F)

##### cbind
ela_sh = read_csv('data/ela_data_sh.csv') %>%
  select(-model_mae) %>%
  select(-type)

ela_normal = read_csv('data/ela_data.csv')%>%
  filter(type == 'TE') %>%
  select(-type)

ela_combined = inner_join(ela_sh, ela_normal, by = c("bench", "dim", "instance", "rep"))
write.csv(ela_combined, 'data/ela_data_combined.csv', row.names = F)

########################################################################################################
######################################################################################
# Read algorithm data and filter down to 100*D budget
cl = makeCluster(detectCores())
registerDoParallel(cl)
aas_results = foreach(i = 1:nrow(bench_info), .combine = rbind) %dopar% {
  library(dplyr)
  library(readr)
  library(stringr)
  library(foreach)
  bench = bench_info[[i, 1]]
  ins = bench_info[[i, 2]]
  meta = filter(meta_data, bname == bench)
  avail = filter(benchmark_files, bname == bench, instance == ins)
  if (avail$avail) {
    tmp_files = aas_files[str_detect(aas_files, paste(bench, ins, '', sep = "_"))]
    
    data = lapply(tmp_files, read_csv) %>%
      bind_rows() %>%
      filter(nfev <= (100 * meta$dim))
    
    data
  }
}
stopCluster(cl)
rm(cl)
registerDoSEQ()

ggplot(aas_results, aes(bname, xx)) +
  geom_boxplot()
