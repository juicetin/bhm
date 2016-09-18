from sklearn.preprocessing import PolynomialFeatures

def poly_features(data, polyspace=2):
    pf = PolynomialFeatures(polyspace)
    return pf.fit_transform(data)
