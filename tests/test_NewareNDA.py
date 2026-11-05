import os
import subprocess
import sys
import tempfile
import pandas as pd
import NewareNDA
from datetime import datetime


def test_NewareNDA(nda_file, ref_file, software_cycle_number, cycle_mode):
    df, _active_mass = NewareNDA.read(nda_file, software_cycle_number, cycle_mode)
    ref_df = pd.read_feather(ref_file)

    # Convert dates to timestamps for comparison
    df['Timestamp'] = df['Timestamp'].apply(datetime.timestamp)
    ref_df['Timestamp'] = ref_df['Timestamp'].apply(datetime.timestamp)

    pd.testing.assert_frame_equal(df, ref_df, check_like=True)


def test_NewareNDAcli(nda_file, ref_file, software_cycle_number, cycle_mode):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, os.path.basename(nda_file))
        subprocess.run(
            [sys.executable, "-m", "NewareNDA", "--format=feather",
             *([] if software_cycle_number else ["--no_software_cycle_number"]),
             f"--cycle_mode={cycle_mode}", nda_file, f"{filename}.ftr"],
            check=True,
        )
        df = pd.read_feather(f"{filename}.ftr")
    ref_df = pd.read_feather(ref_file)

    # Convert dates to timestamps for comparison
    df['Timestamp'] = df['Timestamp'].apply(datetime.timestamp)
    ref_df['Timestamp'] = ref_df['Timestamp'].apply(datetime.timestamp)

    pd.testing.assert_frame_equal(df, ref_df, check_like=True)
