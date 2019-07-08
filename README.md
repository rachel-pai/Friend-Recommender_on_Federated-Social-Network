# Friend recommender on Federated social networks 


Federated social network is popular nowadays because it protects users’ privacy. Friend recommender based on decentralized social network is important and also hard to tackle since there are only limited user information can be retrieved.

In this project, link prediction based on feature-based machine learning method is applied to implement friend recommender system. First, different sampling methods are applied to retrieve the whole social network graph, then two community detection methods, infomap and louvian algorithms, are applied to cut graph into small communities. In each cluster, different similarities, which are regarded as features, are calculated for each vertex. After feature selection, different classifers are trained. Random forest performs best with 96% area under accuracy and 71% area under precisionrecall.

## Sampling mathods
* Breadth FFirst Search 
* Depth First Search 
* Degree Sampling 
* Random Walk 
* Randomly choose vertices 

## Community detection methods
infomap and louvian algorithms

## Similarity metrices
### Neighbor-based
* common neighbors
* Salton Cosine Similarity
* Hub Promoted
* Preferential Attachment
### Path-based 
* Katz
### Random walk based metrics
* Hitting Time
* Commute Time
* Cosine Similarity Time
* simRank
* Rooted PageRank(RPR):
### Social theory based metrics

## Binary classifier
* linear SVM
* logistic regression
* decision tree
* random forest

Classifiers were trained on Spark.
Code was also published on [Databricks](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3472776414846792/37822860499064/5811782744933419/latest.html).

## Detail Methods
See the report for details 


