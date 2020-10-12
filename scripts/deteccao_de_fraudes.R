#### PROBLEMA DE NEGOCIO ####
# Construir um modelo de machine learning cujo preveja se um clique é fraudulento ou nao.
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

#### CLEAN DE MEMORIA E EXLUSAO DE OBJETOS (utilizar somente para alternar entre datasets muito grandes) ####
rm(list = ls(all.names = TRUE));gc()

#### DIRETORIO DE TRABALHO ####
setwd("E:/projetos/deteccao_de_fraudes_no_trafego_de_cliques")

#### LIBRARYS ####
pacman::p_load(tidyverse, caTools, corrplot, caret, data.table,knitr, gridExtra,gmodels, class,e1071,ROCR)

#### CARREGANDO DATASETS ####
list.files()
df <- fread("train.csv")
df2 <- df %>% 
  filter(is_attributed == 1) %>% 
  sample_frac(0.3);dim(df2)

# Para mantermos o balanceamento da nossa amostra, vamos subsetar outro dataset com 137k de observacoes
# considerando is_attibruted == 0

df <- df %>% filter(is_attributed == 0);dim(df)
df <- df[1:137054,];dim(df)
df <- rbind(df,df2);kable(head(df));kable(tail(df));glimpse(df)

#### DICIONARIO DE DADOS ####
# ip: endereço IP de clique
# app: ID do aplicativo
# device: ID do tipo de celular do usuário
# os: ID da versão do celular 
# channel: ID do canal do editor de anúncios
# click_time: data e hora do clique
# attribute_time: horario de dowload do app caso o usuario o tenha baixado depois de clicar no anuncio.
# is_attributed: 1 se o aplicativo foi baixado e 0 se nao foi baixado.

#### PRÉ-PROCESSAMENTO ####
glimpse(df)
colSums(is.na(df)) # Não há valores NA

# Dropando colunas que nao iremos utilizar, e criando novas variáveis
df <- df %>% 
  mutate_if(is.integer,as.factor) %>% 
  select(c(-attributed_time)) %>% 
  mutate_if(is.character,as.POSIXct) %>% 
  mutate(hora_click = as.factor(hour(click_time)),
         dia_click = as.factor(weekdays(click_time))) %>% 
  select(c(-click_time))

#### ANÁLISE EXPLORATÓRIA ####
df3 <- df2 %>% 
  mutate_if(is.integer,as.factor) %>% 
  mutate_if(is.character,as.POSIXct) %>% 
  mutate(hora_click = as.factor(hour(click_time)),
         dia_click = as.factor(weekdays(click_time)),
         hora_attributed = as.factor(hour(attributed_time)),
         dia_attributed = as.factor(weekdays(attributed_time))) %>% 
  select(ip,app,device,os,channel,dia_click,hora_click,dia_attributed,hora_attributed,is_attributed)

p1 <- ggplot(df3, aes(x=dia_click))+
  geom_bar(fill = "lightblue")+theme_minimal()+labs(x = "Dia",title="Click por dia")
p2 <- ggplot(df3, aes(x=dia_attributed))+
  geom_bar(fill = "lightgreen")+theme_minimal()+labs(x = "Dia",title = "Download por dia")
grid.arrange(p1,p2,nrow=2)

# Segunda-feira é o pior dia, tanto quando falamos em downloads quando falamos em clicks.
# É possível notar também que existe um padrão quanto ao comportamento de ambas as amostras.

df_os <- df3 %>% group_by(os) %>%  summarize(n = n()) %>%  arrange(desc(n));df_os <- df_os[1:10,]
df_os_na <- df3 %>% group_by(os) %>%  summarize(n = n()) %>%  arrange(desc(n));df_os <- df_os[1:10,]

p4 <- ggplot(df_os, aes(x = reorder(os,n),y=n))+
  geom_col(fill = "steelblue")+theme_minimal()+labs(x = "os",title = "Numero de dowloads por tipo de os (top 10)")

grid.arrange(p4,nrow=1)

# Os usuário que possuem o os numero 19 e 13 são os que mais efetuam downloads.

p5 <- ggplot(df3, aes(x = hora_click))+
  geom_histogram(stat = "count",fill = "steelblue")+theme_minimal()+labs(x = "Hora",title = "Click por hora")

p6 <- ggplot(df3, aes(x = hora_attributed))+
  geom_histogram(stat = "count",fill = "lightgreen")+theme_minimal()+labs(x = "Hora",title = "Download por hora")

grid.arrange(p5,p6,nrow=1)


# Embora existam algumas diferencas entre o horário de click e de download (especialmente entre a faixa de 0 a 4), 
# pode-se dizer que as amostras possuem o mesmo padrao.

#### DADOS DE TREINO E DE TESTE ####
glimpse(df)
df_final <- df %>% select(1:5,7,8,6) %>% 
  mutate(ip = as.numeric(ip),
         hora_click = as.numeric(hora_click),
         device = as.numeric(device),
         os = as.numeric(os),
         channel = as.numeric(channel),
         is_attributed = as.factor(is_attributed),
         dia_click = as.factor(dia_click),
         app = as.numeric(app));glimpse(df_final)




sample <- sample.split(df_final$ip, SplitRatio = 0.7)
treino <- subset(df_final,sample == TRUE)
teste <- subset(df_final,sample == FALSE)

glimpse(treino);glimpse(teste)


#### SEPARANDO OS ATRIBUTOS E AS CLASSES ####
teste.att <- data.frame(teste[,-8])
teste.class <- data.frame(teste[,8])


#### FEATURE SELECTION ####
formula <- "is_attributed~."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2 )
model <- train(formula, data = treino, method = "glm", trControl = control)
importance <- varImp(model,scale = FALSE)
plot(importance)

#### CONSTRUINDO UM NOVO MODELO COM AS VARIAVEIS SELECIONADAS ####
formula.new <- "is_attributed~app+ip+channel+os"
formula.new <- as.formula(formula.new)
modelo <- glm(formula=formula.new,data=treino,family = "binomial");summary(modelo)

#### PREVENDO E AVALIANDO O MODELO ####
previsoes <- round(predict(modelo, teste, "response"))
previsoes_new <- round(as.data.frame(predict(modelo, teste, "response")))
colnames(previsoes_new) <- "previsoes";previsoes_new$previsoes <- as.factor(previsoes_new$previsoes)

# Confusion matrix
confusionMatrix(table(data = previsoes_new$previsoes,reference = teste.class$is_attributed), positive = "1")


#### PLOT DO MODELO DE MELHOR ACURACIA ####
modelo_final <- modelo
previsoes <- predict(modelo,teste.att,"response")
avaliacao <- prediction(previsoes, teste.class)

plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col = "black",lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8,xaxs = "i", yaxs = "i")
  abline(0,1, col = "red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
  
}

# Plot
par(mfrow = c(1,1))
plot.roc.curve(avaliacao,title.text = "Curva Roc")

