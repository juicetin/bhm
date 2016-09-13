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

### DM4 , with the other using poly features space 2 for features_sn
*LR
For mismatches, from 2nd most probable to least probable compared to det_labels:
    2-th most likely occurrences: 587565    23.022218391052302%
    3-th most likely occurrences: 463442    18.15877891907459%
    4-th most likely occurrences: 1501158   58.81900268987311%
Argmax of the dm distrs and the deterministic labels had: 146005 matches, i.e. 5.411260224522547%

*RF
For mismatches, from 2nd most probable to least probable compared to det_labels:
    2-th most likely occurrences: 1389509   60.49202180920257%
    3-th most likely occurrences: 399720    17.40173756166707%
    4-th most likely occurrences: 507783    22.106240629130365%
Argmax of the dm distrs and the deterministic labels had: 401158 matches, i.e. 14.867780755104384%

*GP
For mismatches, from 2nd most probable to least probable compared to det_labels:
    2-th most likely occurrences: 822996    32.34209377366476%
    3-th most likely occurrences: 565519    22.2237635769665%
    4-th most likely occurrences: 1156144   45.43414264936874%
Argmax of the dm distrs and the deterministic labels had: 153511 matches, i.e. 5.68944877453978%


## Polynomial space 2
* DM is the fixed cluster downsampled version
* DM vs LR
    For mismatches, from 2nd most probable to least probable compared to det_labels:
    0 most likely occurrences: 61312        2.302184206277387%
    1 most likely occurrences: 210390       7.899865200265845%
    2 most likely occurrences: 24302        0.9125078382853775%
    3 most likely occurrences: 40482        1.5200453587963398%
    4 most likely occurrences: 68996        2.5907082055113944%
    5 most likely occurrences: 91085        3.420120831627998%
    6 most likely occurrences: 97134        3.647252751378975%
    7 most likely occurrences: 104892       3.938555352375517%
    8 most likely occurrences: 35636        1.3380844920227846%
    9 most likely occurrences: 46322        1.73932960600178%
    10 most likely occurrences: 42616       1.6001742258402456%
    11 most likely occurrences: 52836       1.9839216584497654%
    12 most likely occurrences: 83136       3.1216464341903194%
    13 most likely occurrences: 114827      4.311601413332031%
    14 most likely occurrences: 161838      6.076802054663357%
    15 most likely occurrences: 236639      8.885480303843858%
    16 most likely occurrences: 194856      7.316584122168361%
    17 most likely occurrences: 197219      7.405311635207138%
    18 most likely occurrences: 187315      7.03342958309709%
    19 most likely occurrences: 128548      4.8268067482474155%
    20 most likely occurrences: 84892       3.187581903041818%
    21 most likely occurrences: 87092       3.270188982468525%
    22 most likely occurrences: 310845      11.67181709290668%
    Argmax of the dm distrs and the deterministic labels had: 34960 matches, i.e. 0.012956930067416064%
* DM vs RF
    For mismatches, from 2nd most probable to least probable compared to det_labels:
    0 most likely occurrences: 104601       4.034593803987729%
    1 most likely occurrences: 39866        1.5376823987320851%
    2 most likely occurrences: 24491        0.9446490650516103%
    3 most likely occurrences: 34050        1.3133518706874907%
    4 most likely occurrences: 75513        2.9126325935748745%
    5 most likely occurrences: 112208       4.328005483292275%
    6 most likely occurrences: 198469       7.655202126974318%
    7 most likely occurrences: 198057       7.639310762195368%
    8 most likely occurrences: 36177        1.3953929699225065%
    9 most likely occurrences: 34543        1.332367508639001%
    10 most likely occurrences: 33903       1.3076818934483991%
    11 most likely occurrences: 37141       1.4325756778033505%
    12 most likely occurrences: 82242       3.1721786945398116%
    13 most likely occurrences: 162597      6.271573395541083%
    14 most likely occurrences: 193475      7.462577185940153%
    15 most likely occurrences: 234039      9.02718233373949%
    16 most likely occurrences: 157836      6.087935561287247%
    17 most likely occurrences: 104497      4.030582391519257%
    18 most likely occurrences: 124297      4.794293611478503%
    19 most likely occurrences: 201107      7.756953147088081%
    20 most likely occurrences: 218005      8.408730530667441%
    21 most likely occurrences: 93214       3.595382709963693%
    22 most likely occurrences: 92275       3.559164283926232%
    Argmax of the dm distrs and the deterministic labels had: 105567 matches, i.e. 0.03912540722044942%

