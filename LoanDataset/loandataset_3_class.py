from __future__ import annotations

import random

from river import datasets

"""
    this is a data generator with data and concept drift with 3 class target where 
    0 means no loan is given
    1 means loan is given under conditions
    2 means loan is given 
"""
class LoanDataset(datasets.base.SyntheticDataset):
    r"""Loan Dataset stream generator based on the generator introduced by Agrawal et al. [^1].

    The generator produces a stream containing eight features, six numeric and two categorical.
    There are 3 functions defined for generating binary class labels from the features.
    Presumably these determine whether the loan should be approved and can be used to simulate
    a concept drift. Moreover, there are 3 functions defined for generating random values for
    the salary variable and can be used to simulate a concept drift.

    **Feature** | **Description** | **Values**

    * `salary` | salary | uniformly distributed from 20k to 150k

    * `commision` | commision | uniformly distributed from 0 to 0.1 x salary

    * `age` | age | uniformly distributed from 20 to 80

    * `educationlevel` | education level | uniformly chosen from 0 to 4

    * `zipcode` | zip code of the town | uniformly chosen from 0 to 8

    * `housevalue` | house value | uniformly distributed from 100k x (8 - zipcode + 1) to 2 x 100k x (8 - zipcode + 1)

    * `loanyears` | years of the loan | uniformly distributed from 10 to 30

    * `loan` | total loan amount | uniformly distributed from 10000 to 10000 x 0.85 x housevalue

    Parameters
    ----------
    seed
        Random seed for reproducibility.

    Notes
    -----
    The sample generation works as follows: The 8 features are generated with the random generator,
    initialized with the seed passed by the user. Then, the classification function decides, as a
    function of all the attributes, whether to classify the instance as class 0 or class 1. The
    next step is to add concept or data drifts using functions generate_concept_drift and
    generate_data_drift, respectively. Three example scenarios are provided below.

    Example scenario 1: An external bank crisis impacts the economy. P(X) stays the same, however
    P(Y|X) changes as the criteria for receiving a loan are stricter. So, one must call
    generate_concept_drift("crisis") to change the classification function.

    Example scenario 2: An internal crisis impacts the economy. P(Y|X) stays the same (bank loan
    criteria do not change), however P(X) changes as there is impact on the salaries. So, one must
    call generate_data_drift("crisis") to change the salary generation function.

    Example scenario 1: A full crisis impacts the economy. P(X) changes as salaries drop and
    P(Y|X) also changes as the criteria for receiving a loan are stricter. So, one must call
    generate_concept_drift("crisis") and generate_data_drift("crisis") to change both the
    classification function and the salary generation function. Note that the order of these
    calls makes no difference.

    Similar scenarios can be simulated for growth or even for diverse situations (e.g. salaries
    rise, however loan criteria remain strict due to some external factor)

    References
    ----------
    [^1]: Agrawal, R., Ghosh, S., Imielinski, T., Iyer, B., & Swami, A. N. (1992).
          An interval classifier for database mining applications. In VLDB (Vol. 92,
          pp. 560-573). Available: https://agrawal-family.com/rakesh/papers/vldb92ic.pdf.

    """

    def __init__(self, seed: int | None = None):
        super().__init__(n_features=8, n_classes=3, n_outputs=1, task=datasets.base.MULTI_CLF)

        # Classification functions to use
        self.classification_function = self._classification_function_normal
        self.salary_function = self._salary_function_normal
        self.seed = seed
        self.n_num_features = 6
        self.n_cat_features = 2
        self._next_class_should_be_zero = False
        self.feature_names = ["salary", "commission", "age", "educationlevel", "zipcode", "housevalue", "loanyears", "loan"]
        self.target_values = [i for i in range(self.n_classes)]

    def __iter__(self):
        self._rng = random.Random(self.seed)
        self._next_class_should_be_zero = False

        while True:
            salary = self.salary_function()
            commission = 0.1 * salary * self._rng.random()
            age = self._rng.randint(20, 80)
            educationlevel = self._rng.randint(0, 4)
            zipcode = self._rng.randint(0, 8)
            housevalue = (8 - zipcode + 1) * 100000 * (1 + self._rng.random())
            loanyears = self._rng.randint(10, 30)
            loan = 10000 + self._rng.random() * 0.75 * housevalue
            y = self.classification_function(salary, commission, age, educationlevel, zipcode, housevalue, loanyears, loan)

            x = dict()
            for feature in self.feature_names:
                x[feature] = eval(feature)

            yield x, y

    def generate_concept_drift(self, event):
        """
        Generates a concept drift by changing the classification function.
        In mathematical terms, P(X) stays the same and P(Y|X) changes.
        """
        if event == "crisis":
            self.classification_function = self._classification_function_crisis
        elif event == "normal":
            self.classification_function = self._classification_function_normal
        elif event == "growth":
            self.classification_function = self._classification_function_growth
        else:
            raise ValueError("event must be one of crisis, normal, growth")

    def generate_data_drift(self, event):
        """
        Generates a data drift by changing the salary function.
        In mathematical terms, P(Y|X) stays the same and P(X) changes.
        """
        if event == "crisis":
            self.salary_function = self._salary_function_crisis
        elif event == "normal":
            self.salary_function = self._salary_function_normal
        elif event == "growth":
            self.salary_function = self._salary_function_growth
        else:
            raise ValueError("event must be one of crisis, normal, growth")

    def _salary_function_growth(self):
        return 30000 + 50000 * self._rng.random()

    def _salary_function_normal(self):
        return 20000 + 40000 * self._rng.random()

    def _salary_function_crisis(self):
        return 10000 + 30000 * self._rng.random()

    def _classification_function_growth(self, salary, commission, age, educationlevel, zipcode, housevalue, loanyears, loan):  # @UnusedVariable
        salaryvar = salary + 1 * commission # this is the expected salary

        if loan > 50 * salaryvar:
            return 0

        if loan > 0.9 * housevalue:
            return 0

        if 35 < age < 45 and loan/loanyears < salaryvar/15 and salaryvar >= 9000:
            return 1
        elif age < 40 and salaryvar >= 15000:
            return 2
        elif age < 20 and salaryvar >= 10000:
            return 1
        elif age < 50 and salaryvar >= 20000 and 0.7*housevalue <= loan:
            return 1
        # elif age < 40 and salaryvar >= 15000:
        #     return 1
        elif age < 50 and salaryvar >= 20000:
            return 2
        elif age < 65 and 12000 <= salaryvar <= 35000:
            return 1
        elif age < 65 and salaryvar >= 35000:
            return 2
        else:
            return 0

    def _classification_function_normal(self, salary, commission, age, educationlevel, zipcode, housevalue, loanyears, loan):  # @UnusedVariable
        salaryvar = salary + 0.5 * commission # this is the expected salary

        if loan > 20 * salaryvar:
            return 0

        if loan > 0.7 * housevalue:
            return 0

        if 35 < age < 45 and loan/loanyears < salaryvar/15 and salaryvar >= 18000:
            return 1
        elif age < 40 and salaryvar >= 25000:
            return 2
        elif age < 20 and salaryvar >= 20000:
            return 1
        elif age < 50 and salaryvar >= 30000 and 0.5*housevalue <= loan:
            return 1
        # elif age < 40 and salaryvar >= 25000:
        #     return 1
        elif age < 50 and salaryvar >= 30000:
            return 2
        elif age < 65 and 20000 <= salaryvar <= 45000:
            return 1
        elif age < 65 and salaryvar >= 45000:
            return 2
        else:
            return 0

    def _classification_function_crisis(self, salary, commission, age, educationlevel, zipcode, housevalue, loanyears, loan):  # @UnusedVariable
        salaryvar = salary # this is the expected salary

        if loan > 10 * salaryvar:
            return 0

        if loan > 0.5 * housevalue:
            return 0

        if 35 < age < 45 and loan/loanyears < salaryvar/15 and salaryvar >= 25000:
            return 1
        elif age < 40 and salaryvar >= 40000:
            return 2
        elif age < 20 and salaryvar >= 30000:
            return 1
        elif age < 50 and salaryvar >= 50000 and 0.35*housevalue <= loan:
            return 1
        # elif age < 40 and salaryvar >= 40000:
        #     return 1
        elif age < 50 and salaryvar >= 50000:
            return 2
        elif age < 65 and 35000 <= salaryvar <= 60000:
            return 1
        elif age < 65 and salaryvar >= 60000:
            return 2
        else:
            return 0
