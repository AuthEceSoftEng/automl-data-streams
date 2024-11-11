from LoanDataset.loandataset_2_class import LoanDataset as LoanDataset_2_class
from LoanDataset.loandataset_3_class import LoanDataset as LoanDataset_3_class
from LoanDataset.loandataset_4_class import LoanDataset as LoanDataset_4_class


def create_loandataset(class_num, datalimit, conceptdriftpoints: dict, datadriftpoints: dict, seed):
    """
            Create the evaluation for every model with the metric of our decision

            Args:
                class_num: The number of class in the output, it can be 2, 3 or 4
                datalimit: The number of the samples in dataset
                conceptdriftpoints: The points of data drift and the kind of drift
                datadriftpoints: The points of consept drift and the kind of drift
                seed: Seed of the generator

            returns:
                data: The dataset ready for the model

        """
    LoanDataset = None
    match class_num:
        case 2:
            LoanDataset = LoanDataset_2_class
        case 3:
            LoanDataset = LoanDataset_3_class
        case 4:
            LoanDataset = LoanDataset_4_class

    # Load loan dataset
    dataset = LoanDataset(seed)
    print(dataset)
    data = []

    for i, (x, y) in enumerate(dataset.take(datalimit)):
        if i in conceptdriftpoints:
            dataset.generate_concept_drift(conceptdriftpoints[i])
        if i in datadriftpoints:
            dataset.generate_data_drift(datadriftpoints[i])
        data.append({**x, "y": y})

    return data
