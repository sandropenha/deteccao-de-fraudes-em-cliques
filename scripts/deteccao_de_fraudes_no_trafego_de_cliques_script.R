#### LIBRARYS ####

## Carrega os pacotes na memória do Rstúdio:
pacman::p_load(tidyverse, caTools, corrplot, caret, data.table,knitr, gridExtra,gmodels, 
               class,e1071,ROCR,DMwR,fasttime,lubridate)


#### DATASET ####

## Carrega o dataset:
df <- as_tibble(fread("train.csv"));df

## Verifica se há valores NA no dataset:
colSums(is.na(df))


#### DICIONARIO DE DADOS ####

## Recursos das variáveis:
# ip: endereco ip do click
# app: id do app utilizado
# device: id do tipo de modelo que o usuário está utilizando (ex: iphone 6, iphone 7, etc.)
# os: id da versao do sistema operacional que o usuário está utilizando
# channel: id do canal mobile da propaganda
# click_time: horario do click (UTC)
# attributed_time: horario do download, caso o usuário tenha o efetuado.
# is_attributed: variavel target. Indica se o app foi baixado ou nao. 0 = nao, 1 = sim.

## Obs do dataset:
# ip, app, device, os e channel possuem encode.


#### AED - DADOS AMOSTRAIS ####

## Ajuste dos dados
df <- df %>% 
  sample_frac(0.10) %>%  # Coleta amostra aleatória de 10% dos dados.
  mutate(click_time = fastPOSIXct(click_time),
         hora_click = hour(click_time),
         dia_click = weekdays(click_time),
         is_attributed = factor(is_attributed, levels = c("0","1"), labels = c("no","yes")),
         attributed_time = ifelse(attributed_time == "",NA,attributed_time),
         attributed_time = fastPOSIXct(attributed_time),
         hora_attributed = hour(attributed_time),
         dia_attributed = weekdays(attributed_time)) %>% 
  select(ip,app,device,os,channel,click_time,hora_click,dia_click,
         attributed_time,hora_attributed,dia_attributed,is_attributed)


## Downloads por ip, app, device e os
p1 <- df %>% filter(is_attributed == "yes") %>% group_by(os) %>% summarize(n = n()) %>% arrange(desc(n)) %>% 
  head() %>%
  ggplot(aes(x=reorder(os,n),y=n))+
  geom_col(fill = "steelblue")+theme_minimal()+labs(x="OS",y="Count",title="Top downloads por OS")+
  geom_text(aes(label = round(n / sum(n), 2)), vjust = 1.6, color = "white", size=3.5)

p2 <- df %>% filter(is_attributed == "yes") %>% group_by(ip) %>% summarize(n = n()) %>% arrange(desc(n)) %>% 
  head() %>% 
  ggplot(aes(x=reorder(ip,n),y=n))+
  geom_col(fill = "steelblue")+theme_minimal()+labs(x="IP",y="Count",title="Top downloads po IP")+
  geom_text(aes(label=round(n/sum(n),2)),vjust=1.6,color="white",size = 3.5)

p3 <- df %>% filter(is_attributed == "yes") %>% group_by(app) %>% summarize(n = n()) %>% arrange(desc(n)) %>% 
  head() %>% 
  ggplot(aes(x=reorder(app,n),y=n))+
  geom_col(fill = "steelblue")+theme_minimal()+labs(x="APP",y="Count",title="Top downloads po APP")+
  geom_text(aes(label=round(n/sum(n),2)),vjust=1.6,color="white",size = 3.5)

p4 <- df %>% filter(is_attributed == "yes") %>% group_by(device) %>% summarize(n = n()) %>% arrange(desc(n)) %>% 
  head() %>% 
  ggplot(aes(x=reorder(device,n),y=n))+
  geom_col(fill="steelblue")+theme_minimal()+labs(x="Device",y="Count",title="Top downloads por Device")+
  geom_text(aes(label=round(n/sum(n),2)),vjust=1.6,color="white",size=3.5)

gridExtra::grid.arrange(p1,p2,p3,p4,ncol=2,nrow=2)


## Downloads e clicks por dia
options(scipen = 999)
p5 <- df %>% filter(is_attributed == "yes") %>% group_by(dia_attributed) %>%
  summarize(n=n()) %>% arrange(desc(n)) %>% 
  ggplot(aes(x=reorder(dia_attributed,n),y=n))+
  geom_col(fill="steelblue")+theme_minimal()+labs(x="Dia",y="Count",title="Taxa de donwloads por dia da semana")+
  geom_text(aes(label=round(n/sum(n),2)),vjust=1.6,color="white",size=3.5)

p6 <- df %>% filter(is_attributed == "no") %>% group_by(dia_click) %>% summarize(n=n()) %>% arrange(desc(n)) %>% 
  ggplot(aes(x=reorder(dia_click,n),y=n))+
  geom_col(fill="steelblue")+theme_minimal()+labs(x="Dia",y="Count",title="Taxa de clicks por dia da semana")+
  geom_text(aes(label=round(n/sum(n),2)),vjust=1.6,color="white",size=3.5)

gridExtra::grid.arrange(p5,p6,nrow=2)

## Downloads e clicks por hora
p7 <- df %>% filter(is_attributed == "no")  %>% group_by(hora_click) %>% 
  summarize(n=n()) %>% arrange(desc(n)) %>% 
  ggplot(aes(x=hora_click,y=n))+
  geom_col(fill="steelblue")+theme_minimal()+labs(x="hora",y="Count", title="Clicks por hora")

p8 <- df %>% filter(is_attributed == "yes") %>%
  group_by(hora_attributed) %>% 
  summarize(n=n()) %>% arrange(desc(n)) %>% 
  ggplot(aes(x=hora_attributed,y=n))+
  geom_col(fill="steelblue")+theme_minimal()+labs(x="hora",y="Count",title="Download por hora")

gridExtra::grid.arrange(p7,p8,nrow=2)



#### MODELO PREDITIVO ####

## Carregando população
rm(list = ls(all.names = TRUE));gc()
pacman::p_load(tidyverse, caTools, corrplot, caret, data.table,knitr, gridExtra,gmodels, 
               class,e1071,ROCR,DMwR,fasttime,lubridate,xgboost)

df <- as_tibble(fread("train.csv"))

## Balanceamento da variável target
table(df$is_attributed)
df2 <- df %>% dplyr::filter(is_attributed == 1) %>% sample_n(456846);df2
df <-  df %>% dplyr::filter(is_attributed == 0) %>% sample_n(456846);df
df <- rbind(df,df2)
rm(df2)

## Ajuste dos dados
df <- df %>% 
  mutate(click_time = fastPOSIXct(click_time),
         hora_click = hour(click_time),
         dia_click = day(click_time),
         is_attributed = factor(is_attributed, levels = c("0","1"), labels = c("no","yes"))) %>% 
  dplyr::select(ip,app,device,os,channel,hora_click,dia_click,is_attributed)


## Salva os dados ajustados
write.csv(df, "dados_ajustados.csv")


## Dados de treino e de teste
treino <- df %>% sample_frac(0.7);treino
teste <- df %>% sample_frac(0.3);teste

## Checando o balanceamento dos dados:
prop.table(table(treino$is_attributed))
prop.table(table(teste$is_attributed))

## Modelo 1 - Naive Bayes:
set.seed(123)
naive1 <- naiveBayes(x=treino[-8],y=treino$is_attributed)
previsao_naive <- predict(naive1,teste[-8])
conf.matrix <- table(teste$is_attributed,previsao_naive)
confusionMatrix(conf.matrix) # 74%

## Modelo 2 - Regressao Logistica
controle <- trainControl(method = "repeatedcv",number = 10,repeats = 2)
regr1 <- train(is_attributed~.,data=treino,method="glm",trControl=controle)
importance <- varImp(regr1,scale=FALSE);plot(importance)

previsoes_reg <- predict(regr1,teste[-8])
conf.matrix <- table(previsoes_reg,teste$is_attributed)
confusionMatrix(conf.matrix) # 77%

## Modelo 3 - Regressao Logística com seleção de variáveis
treino2 <- treino %>% select(-device,-os)
teste2 <- teste %>% select(-device,-os)

controle <- trainControl(method = "repeatedcv",number = 10,repeats = 2)
regr2 <- train(is_attributed~.,data=treino2,method="glm",trControl=controle)

previsoes_reg2 <- predict(regr2,teste2[-6])
conf.matrix <- table(previsoes_reg2,teste2$is_attributed)
confusionMatrix(conf.matrix) # 77%


## Modelo 4 - XGBoost
treino3 <- treino %>% mutate(is_attributed = factor(is_attributed, levels = c("no","yes"), labels = c("0","1")))
teste3 <- teste %>% mutate(is_attributed = factor(is_attributed, levels = c("no","yes"), labels = c("0","1")))

model_xgboost <- xgboost(
  data      = as.matrix(treino3 %>% select(-is_attributed)), # Define as variáveis preditoras.
  label     = as.matrix(treino3$is_attributed), # Define a variável target.
  max.depth = 20,                            # Define o tamanho máximo da árvore.
  eta       = 1,                             # Define a taxa de aprendizado do modelo.
  nthread   = 4,                             # Define o número de threads que devem ser usadas. 
  # Quanto maior for esse número, mais rápido será o treinamento.
  nrounds   = 100,                           # Define o número de iterações.
  objective = "binary:logistic",             # Define que o modelo deve ser baseado em uma regressão logistica binária.
  verbose   = F                              # Exibe a queda da taxa de erro durante o treinamento.
)

previsoes <- predict(model_xgboost,as.matrix(teste3 %>% select(-is_attributed)));length(previsoes);head(previsoes)
previsoes <- as.numeric(previsoes>0.5);head(previsoes)

erros <- mean(previsoes!=teste3$is_attributed)
print(erros)

confusionMatrix(table(pred = previsoes, data = teste3$is_attributed)) # 97%


