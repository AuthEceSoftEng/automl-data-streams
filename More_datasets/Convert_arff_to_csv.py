import pandas as pd
import arff


def convert_arff_to_csv(arff_file, csv_name):
    """
            converts an arff filt to csv

            Args:
                arff_file: path for arff file
                csv_name: path for new csv file

            returns: The new csv file

        """

    # load the arff file and convert it to dataframe, also replace strings with numbers
    df = pd.DataFrame(arff.load(open(arff_file, "r"), encode_nominal=True)["data"])

    # save the df as a csv file
    df.to_csv(csv_name, index=False)
