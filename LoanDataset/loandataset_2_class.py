from __future__ import annotations

import random

from river import datasets

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

    Examples
    --------

    >>> from LoanDataset.loandataset import LoanDataset

    >>> dataset = LoanDataset(seed=42)

    >>> dataset
    Synthetic data generator
    <BLANKLINE>
        Name  LoanDataset
        Task  Binary classification
     Samples  ∞
    Features  8
     Outputs  1
     Classes  2
      Sparse  False
    <BLANKLINE>
    Configuration
    -------------
                       seed  42

    >>> for x, y in dataset.take(5):
    ...     print(list(x.values()), y)
    [45577.07193831535, 113.99169900150872, 37, 1, 3, 683722.7571150863, 13, 393273.6133872848] 1
    [55687.18270819382, 484.137865707138, 47, 0, 0, 984325.7158754332, 17, 422819.0748737112] 1
    [21061.438787354546, 418.7807008558233, 61, 4, 6, 366132.186612209, 28, 86576.38645473785] 0
    [54772.01283171736, 4156.140684923328, 30, 3, 5, 511148.53666865674, 16, 415886.35196781467] 0
    [33463.781804505066, 310.36266661480704, 26, 2, 5, 641490.4125467564, 11, 397898.5531604941] 1

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
        super().__init__(n_features=8, n_classes=2, n_outputs=1, task=datasets.base.BINARY_CLF)

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

        if age < 20 and salaryvar >= 10000:
            return 1
        elif age < 40 and salaryvar >= 15000:
            return 1
        elif age < 60 and salaryvar >= 20000:
            return 1
        else:
            return 0

    def _classification_function_normal(self, salary, commission, age, educationlevel, zipcode, housevalue, loanyears, loan):  # @UnusedVariable
        salaryvar = salary + 0.5 * commission # this is the expected salary

        if loan > 20 * salaryvar:
            return 0

        if loan > 0.7 * housevalue:
            return 0

        if age < 20 and salaryvar >= 20000:
            return 1
        elif age < 40 and salaryvar >= 25000:
            return 1
        elif age < 60 and salaryvar >= 30000:
            return 1
        else:
            return 0

    def _classification_function_crisis(self, salary, commission, age, educationlevel, zipcode, housevalue, loanyears, loan):  # @UnusedVariable
        salaryvar = salary # this is the expected salary

        if loan > 10 * salaryvar:
            return 0

        if loan > 0.5 * housevalue:
            return 0

        if age < 20 and salaryvar >= 30000:
            return 1
        elif age < 40 and salaryvar >= 40000:
            return 1
        elif age < 60 and salaryvar >= 50000:
            return 1
        else:
            return 0
