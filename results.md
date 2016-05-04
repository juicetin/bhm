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
