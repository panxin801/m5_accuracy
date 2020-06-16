class CustomTimeSeriesSpliter:
    def __init__(self, n_splits=5, train_days=365, test_days=28, day_col="d", pred_days=28):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.day_col = day_col
        self.pred_days = pred_days

    def split(self, x, y=None, groups=None):
        sec_in_day = 3600 * 24
        sesc = (x[self.day_col] - x[self.day_col].iloc[0]) * sec_in_day
        duration = sesc.max()

        train_sec = self.train_days * sec_in_day
        test_sec = self.test_days * sec_in_day
        total_sec = train_sec + test_sec
        if self.n_splits == 1:
            train_start = duration - total_sec
            train_end = train_start + train_sec
            train_mask = (sesc >= train_start) & (sesc < train_end)
            test_mask = sesc >= train_end
            yield sesc[train_mask].index.values, sesc[test_mask].index.values
        else:
            step = self.pred_days / sec_in_day
            for nj in range(self.n_splits):
                shift = (self.n_splits - nj + 1) * step
                train_start = duration - total_sec - shift
                train_end = train_start + train_sec
                test_end = train_end + test_sec
                train_mask = (sesc > train_start) & (sesc <= train_end)
                if nj == self.n_splits - 1:
                    test_mask = sesc > train_end
                else:
                    test_mask = (sesc > train_end) & (sesc <= test_end)
                yield sesc[train_mask].index.values, sesc[test_mask].index.values

    def get_splits(self):
        return self.n_splits
