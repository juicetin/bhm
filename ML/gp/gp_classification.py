from ML.gp.gp_regression import GPRegressor

# NOTE
# to call regressor's predict, in this predict, do:
# def predict(...):
#   super(GPClassifier, self).predict(...)

class GPClassifier(GPRegressor):
    def __init__(self):
        pass

    # Unpacks arguments to deal with list of length scales in list of arguments
    def unpack_LLOO_args(self, args):
        f_err = float(args[0])
        l_scales = args[1:self.X.shape[1]+1]
        n_err = args[self.X.shape[1]+1]
        a = float(args[self.X.shape[1]+2])
        b = float(args[self.X.shape[1]+3])
        return f_err, l_scales, n_err, a, b

    def LLOO(self, args):
        f_err, l_scales, n_err, a, b = self.unpack_LLOO_args(args)

        L = self.L_create(self.X, f_err, l_scales, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        mu = self.y - alpha/np.diag(K_inv)
        sigma_sq = 1/np.diag(K_inv)

        LLOO = -sum(norm.cdf(
            self.y * (a * mu + b) /
            np.sqrt(1 + a**2 * sigma_sq)
        ))

        return LLOO

    # NOTE likely incorrect - currently doesn't recalculate K per datapoint
    def LLOO_der(self, args):
        d1 = datetime.now()
        
        f_err, l_scales, n_err, a, b = self.unpack_LLOO_args(args)

        # This block is common to both LLOO and LLOO_der
        L = self.L_create(self.X, f_err, l_scales, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, self.y))) # save for use with derivative func
        K_inv = np.linalg.inv(L.T).dot(np.linalg.inv(L))
        mu = self.y - alpha/np.diag(K_inv)
        sigma_sq = 1/np.diag(K_inv)

        r = a * mu + b

        K_i_diag = np.diag(K_inv)
        dK_dthetas = self.eval_dK_dthetas(f_err, l_scales, n_err) 
        Zs = np.array([K_inv.dot(dK_dtheta) for dK_dtheta in dK_dthetas])
        dvar_dthetas = [np.diag(Z.dot(K_inv))/K_i_diag**2 for Z in Zs] 
        dmu_dthetas = [Z.dot(alpha) / K_i_diag - alpha * dvar_dtheta for Z, dvar_dtheta in zip(Zs, dvar_dthetas)]

        pdf_on_cdf = norm.pdf(r) / norm.cdf(self.y * r)

        # Dervative over LLOO for each of the hyperparameters
        dLLOO_dthetas = [
                -sum(pdf_on_cdf * 
                    (self.y * a / np.sqrt(1 + a**2 * sigma_sq)) * 
                    (dmu_dtheta - 0.5 * a * (a * mu + b) / (1 + a**2 * sigma_sq) * dvar_dtheta))
                    for dmu_dtheta, dvar_dtheta in zip(dmu_dthetas, dvar_dthetas)
        ]

        # Derivative of LLOO w.r.t b
        dLLOO_db_arr = (
            pdf_on_cdf *
            self.y / np.sqrt(1 + a**2 * sigma_sq)
        )
        dLLOO_db = -sum(dLLOO_db_arr)

        # Derivative of LLOO w.r.t a, utilising dLLOO/db
        dLLOO_da = -sum(dLLOO_db_arr *
                        (mu - b * a * sigma_sq) /
                        (1 + a**2 * sigma_sq)
                       )
        
        gradients = dLLOO_dthetas + [dLLOO_da] + [dLLOO_db]

        return np.array(gradients, dtype=np.float64)

    def fit_classes_OvO(self, X, y):
        uniq_y = np.unique(y)
        ovos = combinations(uniq_y, 2)
        self.ovo_pairs = [pair for pair in ovos]
        for class_pair in self.ovo_pairs:
            # Get the two classes involved in this OvO
            pos_class, neg_class = class_pair

            # f_err, l_scales (for each dimension), n_err, alpha, beta
            x0 = [1] + [1] * X.shape[1] + [1,1,1]

            # Set binary labels for each OvO classifier
            cur_idxs = np.where((y != pos_class) & (y != neg_class))
            self.y = y[cur_idxs]
            self.y[np.where(self.y == neg_class)] = -1
            self.y[np.where(self.y == pos_class)] = 1
            self.X = X[cur_idxs]

            # Optimise
            res = minimize(self.LLOO, x0, method='bfgs', jac=self.LLOO_der)

            # Set params for the ibnary OvO
            self.classifier_params[class_pair] = res['x']

            # Reset ys
            self.y = y
            self.X = X

    def fit_classes_OvR(self, X, y):
        uniq_y = np.unique(y)
        prog_bar = Bar('Classes fitted', max=uniq_y.shape[0])
        for c in uniq_y:

            # f_err, l_scales (for each dimension), n_err, alpha, beta
            x0 = [1] + [1] * X.shape[1] + [1, 1, 1]

            # Set binary labels for OvA classifier
            self.y = np.array([1 if label == c else 0 for label in y])

            # Optimise and save hyper/parameters for current binary class pair
            # res = minimize(self.LLOO, x0, method='bfgs')
            res = minimize(self.LLOO, x0, method='bfgs', jac=self.LLOO_der)

            # Set params for current binary regressor (classifier)
            # for param, val in zip(params, res['x']):
            #     self.classifier_params[c][param] = val
            self.classifier_params[c] = res['x']

            # Reset ys
            self.y = y
            prog_bar.next()
        prog_bar.finish()

    # Classification
    def fit_classification(self, X, y):

        self.size = len(y)
        self.X = X
        self.classifier_params = OrderedDict()
        self.class_count = np.unique(y).shape[0]
        # self.class_count = 4
        params = ['f_err', 'l_scales', 'n_err', 'a', 'b']

        # Build OvA classifier for each unique class in y
        # OvR here - also TODO an OvO!
        self.fit_all_classes(X, y)


    # The 'extra' y_ parameter is to allow restriction of y for parallelisation
    def predict_class(self, x, keep_probs=False):

        # TODO vectorize 
        # Vectorize calculation of predictions
        # vec_pred_class_single = np.vectorize(self.predict_class_single)

        # Generate squashed y precidtions in steps
        if x.shape[0] > 5000:
            y_preds = np.zeros((len(self.classifier_params), 2, x.shape[0]))
            step = 2000

            # Step through data and predict in chunks
            for start in range(0, x.shape[0], step):
                next_idx = start + 2000
                end = next_idx if next_idx <= x.shape[0] else x.shape[0]
                cur_preds = np.array([self.predict_class_single(x[start:end], label, params)
                             for label, params in self.classifier_params.items()])
                y_preds[:,:,start:end] = cur_preds

        # Predict smaller datasets all at once
        else:
            y_preds = np.array([
                self.predict_class_single(x, label, params)
                for label, params in self.classifier_params.items()
            ])

        # Unpack means, variances
        y_means, y_vars = y_preds[:,0], y_preds[:,1]

        # Return raw OvA squashed probabilities per class 
        #   (for AUROC scores and GP ensemble methods - PoE, BCM, etc.)
        if keep_probs == True:
            return y_means, y_vars

        y_means_squashed = sigmoid(y_means)

        # Return max squashed value for each data point representing class prediction
        return self.predict_probs(y_means)

    def predict_probs_OvR(self, y_preds):
        return np.argmax(y_preds, axis=0)

    def predict_probs_OvO(self, y_preds):

        # Round each row off to 1s and 0s
        y_rnd = np.abs(np.rint(y_preds).astype(np.int))

        # Convert each row into actual predicted class labels based on OvO pairs
        for row_idx in range(y_rnd.shape[0]):
            # idxs have to be cached first as inlining them will overlap - 
            #    e.g. set to 0, then detected as 0 again
            yes_idxs = np.where(y_rnd[row_idx] == 1)
            no_idxs = np.where(y_rnd[row_idx] == 0)

            y_rnd[row_idx][yes_idxs] = self.ovo_pairs[row_idx][0]
            y_rnd[row_idx][no_idxs] = self.ovo_pairs[row_idx][1]

        # TODO take the max count for each column as class predictions
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=y_rnd)

    # Predict regression values in the binary class case
    def predict_class_single(self, x, label, params):
        # Set parameters
        f_err, l_scales, n_err = self.unpack_GP_args(params)

        # Set y to binary one vs. all labels
        y_ = np.copy(self.y)
        y_[np.where(y_ != label)[0]] = -1
        y_[np.where(y_ != -1)[0]] = 1

        # Set L and alpha matrices
        L = self.L_create(self.X, f_err, l_scales, n_err)
        alpha = linalg.solve(L.T, (linalg.solve(L, y_)))

        # Get predictions of resulting mean and variances
        y_pred, y_var = self.predict_regression(x, L, alpha, f_err, l_scales)

        return y_pred, y_var
