# Algorithms and real dataset
|      Algorithm      |       10F-CV F1      |      10F-CV ACC     |                 Parameters                |  Whitening  |
|:-------------------:|:--------------------:|:-------------------:|:-----------------------------------------:|:-----------:|
|         KNN         |  0.04823190144355188 |  0.1419224345208597 |                  k_n = 5                  |             |
| Logistic Regression | 0.009003066060487903 | 0.17119185801075565 |                                           |             |
| Logistic Regression | 0.009062553606767254 |  0.1711906099150194 | multi_class='multinomial', solver='lbfgs' |             |
|    Random Forest    |  0.04959629830094382 | 0.15240680579263258 |                                           |             |
|         SVC         |  0.00890036952350639 | 0.17119127067158563 |                    OvA                    |             |
|         KNN         |  0.03527194333788781 | 0.11398630765559899 |                  k_n = 5                  | norm, scale |
| Logistic Regression | 0.009056450222784309 | 0.17112967347613017 |                                           | norm, scale |
| Logistic Regression | 0.009149911609957848 | 0.17094869959436892 | multi_class='multinomial', solver='lbfgs' | norm, scale |
|    Random Forest    |  0.05174641763593827 | 0.15543637464897303 |                                           | norm, scale |
|         SVC         | 0.009094924719285665 |  0.1711300405631114 |                    OvA                    | norm, scale |

* norm - normaliz(s)e
* scale - zero mean, 1 variance

# Performance related

## Cache size with time taken to do partial derivatives
X, y = datasets.make_classification(n_samples=100,
        n_features=5, 
        n_clusters_per_class=2,
        n_redundant=0, 
        n_repeated=0,
        n_informative=5,
        n_classes=5)
| Cache size         | Generated Data Parameters   | Derivations (m)   | Lambdafication (m)   | dK_dtheta eval (m)  |
| :----------------: | :-------------------------: | :---------------: | :------------------: | :-----------------: |
| 1000 (default)     | As above, cProfile mode     | 12:42.67          | Untimed              |
| 5000               | As above, cProfile mode     | 09:49:49          | 04:22:96             |
| 10000              | As above, cProfile mode     | 07:29:30          | 04:16:80             |
| 40000              | As above, cProfile mode     | 07.57:57          | 04:02:76             |
| None (unbounded)   | As above, cProfile mode     | 05:42.19          | 03:40:40             |
| 40000              | As above                    | 05:22:63          | 02:47:13             | 00:24:47            |
| 60000              | As above                    | 05:22:53          | 02:47:62             | 00:24:14            |
| 100000             | As above                    | 05:08:81          | 02:47:32             | About the same      |
| 150000             | As above                    | 05:02:14          | 02:47:30             | About the same      |
| 200000             | As above                    | 03:35:73          | 02:54:68             | About the same      |
| 250000             | As above                    | 04:54:82          | 02:48:20             | About the same      |
| 300000             | As above                    | 04:06:57          | 02:41:26             | About the same      |
| 500000             | As above                    | 04:10:49          | 02:42:23             | About the same      |
| None (unbounded)   | As above                    | 04:04:26          | 02:42:00             | 00:23:94            |

# Class sampling - is even or stratified split of the training data better?
| Total sample size | Split method | Test method | Number of runs | Average AUROC |
|:-----------------:|:------------:|:-----------:|:--------------:|:-------------:|
|        500        |     Even     |    10F-CV   |       10       |    0.74197    |
|        500        |  Stratified  |    10F-CV   |       10       |    0.70983    |
|        1000       |     Even     |    10F-CV   |       100      |       ?       |
|        1000       |  Stratified  |    10F-CV   |       100      |       ?       |
