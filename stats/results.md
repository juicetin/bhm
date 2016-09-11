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
Note that for the 500, 1000 test cases below, they are all using the same 500/1000 respectively.

| Data    | Split | Method | Test   | Runs      | Avg AUROC  | Notes               | Avg F1  |
| :-----: | :---: | :----  | :----: | :-------: | :--------: |                     |         |
| 500     | E     | GP     | 10F-CV | 10        | 0.74197    | Coords not included |
| 500     | S     | GP     | 10F-CV | 10        | 0.70983    | Coords not included |
| 500     | E     | GP     | 10F-CV | 1         | 0.82214    | Coords included     |
| 500     | S     | GP     | 10F-CV | 1         | 0.77466    | Coords included vvv |
| 500     | E     | GP     | 10F-CV | 10        | 0.86534    |                     |
| 500     | S     | GP     | 10F-CV | 10        | 0.80136    |                     |
| 1000    | E     | GP     | 10F-CV | 100       | ?          |
| 1000    | S     | GP     | 10F-CV | 100       | ?          |
| 1000    | E     | GP     | All    | 1         | 0.87626    | Deterministic       | 0.56208 |
| 1000    | E     | PoEGP  | All    | 5         | 0.80973    |                     | 0.47481 |
| 1000    | E     | PoEGP  | All    | 200       | 0.80186    |                     | 0.47595 |
| 1000    | E     | GPoEGP | All    | 5         | 0.80864    |                     | 0.51018 |
| 1000    | E     | GPoEGP | All    | 200       | 0.80105    |                     | 0.47748 |
| 1000    | E     | BCM    | All    | 5         | 0.80682    |                     | 0.48167 |
| 1000    | E     | BCM    | All    | 200       | 0.80421    |                     | 0.48227 |
| 1000    | E     | GPy    | All    | 1         | 0.87638    | RBF, EP (default)   | 0.57013 |

Key:
E = even
S = stratified

* BCM* ones will need fixing as 'prior precision' is currently defined incorrectly (?)

| Total sample size   | Split method   | Method   | Test method   | No. runs    | Average AUROC                           | Notes                                          | F1-Score                                |
| :-----------------: | :------------: | :------: | :-----------: | :---------: | :-------------:                         | ---------------------                          | :--------:                              |
| 1000                | Even           | PoEGP    | All points    | 4           | 0.76736,0.76945,0.77309,0.78124         | Expert size: 200, points in each expert random | 0.64515,0.62176,0.74395,0.47536         |
| 1000                | Even           | GPoEGP   | All points    | 4           | 0.78657,0.75807,0.77583,0.79221         | Expert size: 200, ditto                        | 0.32996,0.25872,0.29148,0.32702         |
| 1000                | Even           | BCM      | All points    | 5           | 0.79755,0.79548,0.78110,0.79423,0.77881 | Ditto                                          | 0.55378,0.77672,0.56987,0.43165,0.23686 |

    NOTE best for simple classes - scaling, then normalising features
    order: original, normalised, scaled, normalised-scaled, scaled-normalised
# [0.13911784653850162,   0.62549115824070989,  0.80419877283841434,  0.80067072027980024, 0.77420703810759661, 
#  0.0014678899082568807, 0.017750714162373553, 0.012602890116646892, 0.02482014086796169, 0.029004894912527762]
feature_perms = [features, features_n, features_s, features_ns, features_sn]

[0.74880756856008601, 0.67981634815766179, 0.70133102810814296]
[0.72739502569246961, 0.71183799527284519, 0.65632134135368103]
feature_perms = [features_s, features_ns, features_sn]


# GP vs DM multilabels
## simplified classes
GP  bincount: array([  81855,  454609,  704545, 1457161])
DM1 bincount: array([  27730,  505781,  615009, 1549650]) (all points)
DM2 bincount: array([ 475263,  382189, 1103195,  737523]) (hc clusters)
DM3 bincount: array([ 255393,  376389,  731513, 1334875]) (fixed clusters)
DM4 bincount: array([ 297410,  379470,  874949, 1146341]) (fixed clusters, poly-space 2)
matching labels between the GP,DM1: 1582175/2698170 (56.54320502328307%)
matching labels between the GP,DM2: 1706038/2698170 (63.22944810742096%)
matching labels between the GP,DM3: 1749599/2698170 (64.84391272603283%)
matching labels between the GP,DM3: 2087761/2698170 (77.37692584233017%)
### DM4
* vs GP
    For mismatches, from 2nd most probable to least probable compared to det_labels:
        2 most likely occurrences: 1735
        3 most likely occurrences: 58080
        4 most likely occurrences: 550594
    Argmax of the dm distrs and the deterministic labels had: 2087761 matches, i.e. 0.7737692584233017%
* vs LR default
    For mismatches, from 2nd most probable to least probable compared to det_labels:
        2 most likely occurrences: 10736
        3 most likely occurrences: 109680
        4 most likely occurrences: 424054
    Argmax of the dm distrs and the deterministic labels had: 2153700 matches, i.e. 0.7982076740902168%
* vs RF default
    For mismatches, from 2nd most probable to least probable compared to det_labels:
        2 most likely occurrences: 901393
        3 most likely occurrences: 683908
        4 most likely occurrences: 463301
    Argmax of the dm distrs and the deterministic labels had: 649568 matches, i.e. 0.2407439116141681%
