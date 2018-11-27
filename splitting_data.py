import numpy as np
import pandas as pd


np.random.seed(7000000)  # Generate the same labels every time, split into train/cv/test set consistently


def get_data_with_labels():
    """
    Loads the data and attaches labels to it.
    :return: A data frame with all the data and a new column "WasDelirious" which indicates whether the subject was
             delirious at some point in the ICU, not that they were delirious at that specific moment.
    """
    df = pd.read_table("confocal_all_patient_phys_data.txt")

    # Get the number of subjects and generate random labels for them based on how many delirious subjects we should have
    # This is a temporary solution to develop the model while we wait for the real labels.
    number_of_subjects = int(df.subject_id.iloc[-1].lstrip("confocal_"))
    print("We have "+str(number_of_subjects)+" subjects in the data set.")
    print("Generating random labels for the subjects...")

    percentage_of_subjects_who_were_delirious = 0.70
    print("Assuming "+str(percentage_of_subjects_who_were_delirious*100)+"% of subjects were delirious...")

    random_labels = np.random.rand(number_of_subjects) < percentage_of_subjects_who_were_delirious
    was_subject_delirious = pd.DataFrame({"subject_id": df.subject_id.unique(), "WasDelirious": random_labels})

    df = df.merge(was_subject_delirious, on="subject_id")

    print("Our labels for each subject are:")
    print(was_subject_delirious)

    return df


def find_number_of_subjects(df):
    """
    :param df: Data frame of all the subject data.
    :return: The subject number for the last subject in the data frame.
    """
    return int(df["subject_id"].iloc[-1].lstrip("confocal_"))


def separate_delirious_subjects(df):
    """
    :param df: Data frame of all subject data.
    :return: A data frame of all delirious subjects, and a second data frame of all other subjects.
    """
    return df.loc[df["WasDelirious"] == True], df.loc[df["WasDelirious"] == False]


def get_subject_numbers(df):
    """
    Get the subject numbers in the data frame.
    :param df: Data frame to get subject numbers from.
    :return: An array of the subject numbers.
    """
    subject_ids = df["subject_id"].unique()
    subject_numbers = []
    for id in subject_ids:
        subject_numbers.append(id.lstrip("confocal_"))
    return subject_numbers


def get_subject_data(df, subject_number):
    """
    Subjects are identified by their subject_id, which is confocal_NUMBER
    :param df: Data frame of all subject data.
    :param subject_number: The value to fill in for NUMBER in confocal_NUMBER to get the subject id.
    :return: A data frame with all the data for the subject.
    """
    subject_data = df.loc[df['subject_id'] == "confocal_" + str(subject_number)]
    return subject_data


def split_subject_numbers(subject_numbers):
    """
    :param df: Data frame of subjects to split up
    :param subject_numbers: Subject numbers for the subjects in df.
    :return: Three arrays of subject numbers meant to be used for the train set, the cv set, and the test set.
    """
    train_set_percentage = 0.6
    cv_set_percentage = 0.2

    # Assign the subjects to the sets at random, following the distribution from the percentages above
    shuffled_subject_numbers = np.random.permutation(subject_numbers).tolist()
    last_train_set_index = int(len(shuffled_subject_numbers)*train_set_percentage)
    last_cv_set_index = last_train_set_index + int(len(shuffled_subject_numbers) * cv_set_percentage)

    train_set_subject_numbers = shuffled_subject_numbers[:last_train_set_index]
    cv_set_subject_numbers = shuffled_subject_numbers[last_train_set_index:last_cv_set_index]
    test_set_subject_numbers = shuffled_subject_numbers[last_cv_set_index:]

    return train_set_subject_numbers, cv_set_subject_numbers, test_set_subject_numbers


def assign_subject_numbers_to_splits(delirious_subject_numbers, non_delirious_subject_numbers):
    """
    :param delirious_subject_numbers: Array of subject numbers of all patients with delirium.
    :param non_delirious_subject_numbers: Array of subject numbers of all patients without delirium.
    :return: Array of the subject numbers that belong in each of the train, cv, and test sets.
             The distribution of delirious and non-delirious patients is the same in all sets.
    """
    train_subject_nums, cv_subject_nums, test_subject_nums = split_subject_numbers(delirious_subject_numbers)
    train_subject_nums2, cv_subject_nums2, test_subject_nums2 = split_subject_numbers(non_delirious_subject_numbers)
    train_subject_nums += train_subject_nums2
    cv_subject_nums += cv_subject_nums2
    test_subject_nums += test_subject_nums2

    return train_subject_nums, cv_subject_nums, test_subject_nums


def get_data_for_splits(df, train_subject_nums, cv_subject_nums, test_subject_nums):
    """
    :param train_subject_nums: Array of the subjects that go in the train set.
    :param cv_subject_nums: Array of the subjects that go in the cv set.
    :param test_subject_nums: Array of the number of subjects that go in the test set.
    :return: Data frame to use as the train, cv, and test set.
    """
    train_data = pd.DataFrame(columns=df.columns)
    for subject_number in train_subject_nums:
        train_data = train_data.append(get_subject_data(df, subject_number), ignore_index=True)
    cv_data = pd.DataFrame(columns=df.columns)
    for subject_number in cv_subject_nums:
        cv_data = cv_data.append(get_subject_data(df, subject_number), ignore_index=True)
    test_data = pd.DataFrame(columns=df.columns)
    for subject_number in test_subject_nums:
        test_data = test_data.append(get_subject_data(df, subject_number), ignore_index=True)
    return train_data, cv_data, test_data


def get_data_split_up():
    """
    Loads all data and splits up the subjects into train, cv, and test sets.
    :return: Three data frames, the training df, cv df, and the testing df.
    """
    all_subject_data = get_data_with_labels()

    number_of_subjects = find_number_of_subjects(all_subject_data)

    # Identify what subjects have delirium
    delirious_subjects, non_delirious_subjects = separate_delirious_subjects(all_subject_data)
    delirious_subject_numbers = get_subject_numbers(delirious_subjects)
    non_delirious_subject_numbers = get_subject_numbers(non_delirious_subjects)

    # Split up the patients into train/cv/test sets as subject numbers first, then get the respective data
    train_subject_nums, cv_subject_nums, test_subject_nums = assign_subject_numbers_to_splits(delirious_subject_numbers,
                                                                                              non_delirious_subject_numbers)
    train_data, cv_data, test_data = get_data_for_splits(all_subject_data,
                                                         train_subject_nums,
                                                         cv_subject_nums,
                                                         test_subject_nums)

    return train_data, cv_data, test_data


def main():
    train_data, cv_data, test_data = get_data_split_up()
    print("Training head and tail:")
    print(train_data.head(1))
    print(train_data.tail(1))
    print("CV head and tail:")
    print(cv_data.head(1))
    print(cv_data.tail(1))
    print("Testing head and tail:")
    print(test_data.head(1))
    print(test_data.tail(1))


if __name__ == "__main__":
    main()
