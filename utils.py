"""Copyright (c) 2019 AIT Lab, ETH Zurich, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import os
import csv
import patoolib


submission_files = [
    "config.py",
    "dataset.py",
    "dataset_numpy_to_tfrecord.py",
    "model.py",
    "restore_and_evaluate.py",
    "setup.py",
    "Skeleton.py",
    "training.py",
    "utils.py"
    ]


def create_zip_code_files(output_file='submission_files.zip'):
    patoolib.create_archive(output_file, submission_files)


def create_submission_csv(labels, output_file='submission.csv'):
    with open(output_file, 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Id", "y"])

        for i, label in enumerate(labels):
            writer.writerow([i+1, label+1])

def create_softlabel_csv(soft_labels, output_file='softlabel.csv'):
    with open(output_file, 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Id", "y"])

        for i, soft_label in enumerate(soft_labels):
            writer.writerow([i+1, soft_label])


def create_submission_files(labels, soft_labels, out_dir, out_csv_file, out_code_file, out_soft_file):
    create_submission_csv(labels, os.path.join(out_dir, out_csv_file))
    create_zip_code_files(os.path.join(out_dir, out_code_file))
    create_softlabel_csv(soft_labels, os.path.join(out_dir, out_soft_file) )
