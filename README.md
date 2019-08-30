# [Team Human ensemble]

#### [Member]

- H.W. Jeong (INHA Univ) qhsh9713@gmail.com
- J.M. Park (S.N.U) valor1167@gmail.com
- J.H. Park (S.N.U) jungho.p90@gmail.com
- H.Y. Choi (C.N.U) bigchoi3449@gmail.com

----

#### [What we do]

Dimensionality reduction & feature selection (Fully automatic !)

- **Data**
  - Breast Cancer Wisconsin
  - **[To Do] Another data**
- **Dimensionality reduction**
  - Max-min selection
  - Clustering selection
- **Feature selection**
  - Ranking system
    - Chi square score
    - Laplacian score
    - Fisher score

---

### Requirement

```
pandas == 0.22.0
numpy == 1.17.0
scikit-feature ==1.0.0
scipy == 1.3.1
sklearn >= 0.0
xgboost == 0.9.0

```

---

## Breast Caner Classification

### J.M Park, J.H Park, H.W Jung, H.Y Choi


## Abstract


Study background knowledge of features from "nuclear feature extraction for breast tumor diagnosis(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.707&rep=rep1&type=pdf)". Checked mean, standard error, worst of ten Nuclear Features which used to determine a type of tumor; Malignant or Benign. We tried to maximize the  Sensitivity =(predict Malignant)/(total Malignant). This is because patients should notice their aggression of tumor. If tumors are benign, patients have some time to cope with. On the other side, patients have to undergo an operation as soon as possible. In the case of the health clinic, overreacting is better than staying calm. 

1. Common operations in prominent Kernels

   -1. By using Pandas library, scientists attained correlations with Heat map.

   -2. Select features that their 'Correlation' is under 0.7. 

  		['texture_mean','symmetry_mean','texture_se','area_se','smoothness_se'...]

   -3. Run a Random Forest Classifier from sklearn => ABennett, K. P., &amp; Mangasarian, O. L. (n.d.). ROBUST LINEAR PROGRAMMING
DISCRIMINATION OF TWO LINEARLY INSEPARABLE SETS. Retrieved from
https://pdfs.semanticscholar.org/4c5e/562437ee94fb6e4d60ec559386dd0a4335
13.pdf
De Silva, V., &amp; Carlsson, G. (2004). Topological estimation using witness complexes.
In Eurographics Symposium on Point-Based Graphics. Retrieved from
https://pdfs.semanticscholar.org/957a/afd3a7c8736f286b7638eb89d6db5ed309
b1.pdf
Gu, Q., Li, Z., &amp; Han, J. (n.d.). Generalized Fisher Score for Feature Selection.
Retrieved from https://arxiv.org/pdf/1202.3725.pdf
He, X., Cai, D., &amp; Niyogi, P. (n.d.). Laplacian Score for Feature Selection. Retrieved
from https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdfccurary is beyond 0.95.  

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

#### Random result

![random](/home/bono/Desktop/Bono/deep_server/icim_5th_academy/result/figure/random.png)

#### Max-min selection result (Top2)

![top2](/home/bono/Desktop/Bono/deep_server/icim_5th_academy/result/figure/top2.png)

#### Max-min selection result (Bottom 2)

![bottom](/home/bono/Desktop/Bono/deep_server/icim_5th_academy/result/figure/bottom.png)

#### Clustering selection Result

![cluster](/home/bono/Desktop/Bono/deep_server/icim_5th_academy/result/figure/cluster.png)

----



#### Reference

Bennett, K. P., &amp; Mangasarian, O. L. (n.d.). ROBUST LINEAR PROGRAMMING DISCRIMINATION OF TWO LINEARLY INSEPARABLE SETS. Retrieved from https://pdfs.semanticscholar.org/4c5e/562437ee94fb6e4d60ec559386dd0a4335 13.pdf

De Silva, V., &amp; Carlsson, G. (2004). Topological estimation using witness complexes. In Eurographics Symposium on Point-Based Graphics. Retrieved from https://pdfs.semanticscholar.org/957a/afd3a7c8736f286b7638eb89d6db5ed309 b1.pdf

Gu, Q., Li, Z., &amp; Han, J. (n.d.). Generalized Fisher Score for Feature Selection. Retrieved from https://arxiv.org/pdf/1202.3725.pdf He, X., Cai, D., &amp; Niyogi, P. (n.d.). Laplacian Score for Feature Selection. Retrieved from https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf
