# icim_5th_academy
about breast cancer data's feature selection method (breast cancer wisconsin)


## Breast Caner Classification

### J.M Park, J.H Park, H.W Jung, H.Y Choi


## Abstract


Study background knowledge of features from "nuclear feature extraction for breast tumor diagnosis(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.707&rep=rep1&type=pdf)" paper. Check mean, standard error, worst of ten Nuclear Features which used to determine a type of tumor; Malignant or Benign. We tried to maximize the  Sensitivity =(predict Malignant)/(total Malignant). This is because patients should notice their aggression of tumor. If tumors are benign, patients have some time to cope with. On the other side, patients have to undergo an operation as soon as possible. In the case of the health clinic, overreacting is better than staying calm. 

1. Common operations in prominent Kernels

   -1. By using Pandas library, scientists attained correlations with Heat map.

   -2. Select features that their 'Correlation' is under 0.7. 

  		['texture_mean','symmetry_mean','texture_se','area_se','smoothness_se'...]

   -3. Run a Random Forest Classifier from sklearn => Accurary is beyond 0.95.  

   Therefore, we determined it as meaningless because we do not have enough data to split into train and test.

2. Data augment methods to approach to a strategy for getting enough Train data.

   -1. We already have standard errors so that I perturbed mean and worst data (+,-) => obtain fourfold synthetic data

   -2. If we choose standard error as (N/10) * (standard error), we will able to obtain synthetic data however, we can't 

   Therefore, it isdifficult and also mingingless. There is a labeling problem and it is not much different from noise 

3. Find Dominant Features 

   -1. Ways to reduce dimension : PCA, NMF => Not Challenging

    * we realized that around 10 features work most efficiently.
    
 ## Here is our Main part

###   -2. Find new method to obtain ultimate efficiency => "Max Min Selector(MMS)"

   -3. To use MMS which is amazing, we have to determine Initial feature.

   -4. After visualization data with MDS, we observed several features are gathered at some points

   -5. So, we classify them into a cluster 

###   -6. Make an algorithm to pick a representational feature from cluster.

4. Find Initial Feature 

   -1. Scores from Independency tests(Kai square, Laplacian, Fisher)

     * Laplacian and Fisher's scores are somewhat related.

###   -2. Combine them to attain the most believable Initial Feature => Ranking System.

    ['area_worst','radius_worst','area_mean',perimeter_mean'...]

5. Test

   -1. We found dominant features by its ranks 

   -2. Test several models with top tier feature, lowest feature to make 10 or more features.  

## Result

We realized that MMS is a groundbreaking method to reduce dimensions. Even though we select negligible feature for initial feature, MMS automatically add dominant feature on the very next step. Its accuracy always goes over 0.9 at least. Please comment us if you find room for improvement

