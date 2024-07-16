###### Functions #######
format_pvalue <- function(x){
  # if x < 0.0001, using Scientific notation
  fo <- if (x < 0.0001) '%.1e' else '%.4f' 
  sprintf(fo, x)
}

# format_p_value(0.00001)

to_pt <- function(x){
  sprintf("%.0f%%", 100*x)
}

# to_pt(0.01) -> 1%

get_met4_presences <- function(met4) {
  data_pres <- as.data.frame(ifelse(met4[, -1] > 0, 1, 0))
  cbind(Sample_id = met4$Sample_id, data_pres)
}


plot_prevalences <- function(met4, 
                             sigb1 = SIGB1,  
                             sigb2 = SIGB2, pcr = c(SIGB1_PCR, SIGB2_PCR)) {
  
  pres <- get_met4_presences(met4)[, c(sigb1, sigb2, 'Akkermansia_muciniphila')]
  prevalences <- 100 * colSums(pres) / nrow(pres)
  
  data <- data.frame(SPECIES = names(prevalences), PREVALENCE = prevalences,
                     stringsAsFactors = FALSE) %>%
    mutate(TYPE = ifelse(SPECIES == 'Akkermansia_muciniphila', 'Akk',
                         ifelse(SPECIES %in% sigb1, 'SIG1', 'SIG2'))) %>%
    mutate(PCR = ifelse(SPECIES %in% c(pcr, 'Akkermansia_muciniphila'), 'Y', 'N')) %>%
    arrange(PREVALENCE) %>%
    mutate(SPECIES = gsub('_', ' ', SPECIES, fixed = TRUE))
  data$SPECIES <- factor(data$SPECIES, levels = data$SPECIES)
  
  ggplot(data, aes(x = SPECIES, y = PREVALENCE)) +
    geom_bar(aes(fill = TYPE), stat = 'identity') +
    scale_fill_manual(name = '', values = c(SIG1 = 'orange', SIG2 = 'cornflowerblue', Akk = 'pink')) +
    
    # only show part of point
    geom_point(y = 0, shape = 18, data = data %>% filter(PCR == 'Y'), color = 'darkgray') +
    ylim(0, 100) + ylab('Prevalence (%)') + xlab('') +
    coord_flip() + 
    expand_limits(x = 0) +
    GRAPHICS$THEME +
    theme(axis.text.y = element_text(size = 6, hjust = 1), 
          panel.grid.major.y = element_blank(),
          panel.border = element_blank(), 
          axis.ticks.y = element_blank(),
          legend.position = 'bottom')
  
}

### inner_legend function 
inner_legend <- function() {
  theme(legend.position = c(0.98, 0.98),
        legend.justification = c(1, 1), 
        legend.key.width = unit(1, "lines"), 
        legend.key.height = unit(1, "lines"), 
        plot.margin = unit(c(5, 1, 0.5, 0.5), "lines"))
}