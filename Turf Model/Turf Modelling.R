library(tidyverse)


item_fact <- read_csv("D:/Sainsburys/noreturn_item_fact.csv")
product_dim <- read_csv("D:/Sainsburys/product_dim_new_clusters_New.csv")


item_fact <- item_fact[,c("CUSTOMER_KEY", "PRODUCT_KEY", "ITEM_QUANTITY", "WEEK", "DATE_KEY", "PROMOTION_KEY", "EXTENDED_PRICE")]

item_fact <- left_join(item_fact, product_dim)

customer_sales_groupby <- item_fact %>%
    group_by(CUSTOMER_KEY, CLUSTER) %>%
    summarise(NUMBER_OF_TRANSACTIONS = n(), ITEM_QUANTITY = sum(ITEM_QUANTITY), EXTENDED_PRICE = sum(EXTENDED_PRICE))

################# Data Preparation ####################3


#customer_sales_groupby <- read_csv("D:/Sainsburys/Customer_Sale_Groupby.csv")

#customer_sales_groupby <- left_join(customer_sales_groupby)

options(scipen = 99999)

customer_sales_groupby <- customer_sales_groupby[customer_sales_groupby$CUSTOMER_KEY > 0,]

#########################
cluster <- 1
#########################
customer_sales_groupby <- customer_sales_groupby[customer_sales_groupby$CLUSTER == cluster,]

customer_sales_groupby$ITEM_QUANTITY_WEIGHTS <- customer_sales_groupby$ITEM_QUANTITY / sum(customer_sales_groupby$ITEM_QUANTITY)
customer_sales_groupby$EXTENDED_PRICE_WEIGHTS <- customer_sales_groupby$EXTENDED_PRICE / sum(customer_sales_groupby$EXTENDED_PRICE)
customer_sales_groupby$NUMBER_OF_TRANSACTIONS_WEIGHTS <- customer_sales_groupby$NUMBER_OF_TRANSACTIONS / sum(customer_sales_groupby$NUMBER_OF_TRANSACTIONS)

customer_sales_groupby$AVG_WEIGHTS <- (customer_sales_groupby$ITEM_QUANTITY_WEIGHTS + customer_sales_groupby$EXTENDED_PRICE_WEIGHTS + customer_sales_groupby$NUMBER_OF_TRANSACTIONS_WEIGHTS)/3


quantile(customer_sales_groupby$AVG_WEIGHTS)

customer_sales_groupby <- customer_sales_groupby[,-c(2,3,4,5,6,7,8)]

#customer_sales_groupby <- customer_sales_groupby[,c(1,3,2)]

customer_sales_groupby$AVG_WEIGHTS <- scale(customer_sales_groupby$AVG_WEIGHTS)

agg_item_fact_spread <- read_csv("D:/Sainsburys/Spread Data/Agg_item_fact_spread.csv")

product_dim_req_cluster <- product_dim[product_dim$CLUSTER == cluster,]
product_dim_req_cluster <- na.omit(product_dim_req_cluster)

agg_item_fact_spread_custID <- agg_item_fact_spread[,1]

agg_item_fact_spread <- agg_item_fact_spread[,as.character(unique(product_dim_req_cluster$SKU_NO))]

#agg_item_fact_spread <- agg_item_fact_spread[,-1]

agg_item_fact_spread[agg_item_fact_spread > 0] <- 1

agg_item_fact_spread <- cbind(agg_item_fact_spread_custID, agg_item_fact_spread)

agg_item_fact_spread <- left_join(customer_sales_groupby,agg_item_fact_spread)

rm(agg_item_fact_spread_custID)
rm(customer_sales_groupby)
rm(item_fact)
rm(product_dim)
rm(product_dim_req_cluster)

############# Execute from this line

#write_csv(agg_item_fact_spread, "Turf_data_Cluster_1.csv")
#agg_item_fact_spread <- read_csv("D:/Sainsburys/Turf_data_Cluster_0.csv")

#agg_item_fact_spread <- agg_item_fact_spread[agg_item_fact_spread$AVG_WEIGHTS > 0,]

head(agg_item_fact_spread)

library(turfR)

memory.limit(size = 30000)

mdl <- turf(agg_item_fact_spread, 3, 2)


mdl

#data("turf_ex_data")

#turf(turf_ex_data, 10,3:6)


########################### PCA ##########################

pca_data <- agg_item_fact_spread[,c(3:70)]
mu = colMeans(pca_data)

Xpca <- prcomp(pca_data)

names(Xpca)
Xpca$center
Xpca$rotation

dim(Xpca$x)

biplot(Xpca, scale = 0)


std_dev <- Xpca$sdev
pr_var <- std_dev^2
pr_var[1:10]
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

################## TURF with PCA ###############################

train.data <- data.frame(agg_item_fact_spread[,c(1,2)], Xpca$x)

train.data <- train.data[,1:30]

rm(agg_item_fact_spread)
rm(pca_data)

mdl <- turf(train.data, 27, 3)
