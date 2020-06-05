import pathlib
import re
# from pprint import pprint as print
import sys
import os
import subprocess

regex_oarjob_id = re.compile(r".OAR.(\d+).stdout")

def get_matches_from_file(str_file):
    matches = re.findall(regex_oarjob_id, str_file)
    return matches

def add_matches_to_manualy_relaunched():
    file_path = pathlib.Path(sys.argv[-2])
    manualy_relaunched = pathlib.Path(sys.argv[-1])

    all_matches_already_launched = []
    with open(file_path, 'r') as file:
        file_str = file.read()
    all_matches_already_launched.extend(get_matches_from_file(file_str))

    with open(manualy_relaunched, "a") as file:
        for match in all_matches_already_launched:
            file.write("{}\n".format(match))
            # print(match)

def main():
    file_path = pathlib.Path(sys.argv[-2])
    manualy_relaunched = pathlib.Path(sys.argv[-1])

    with open(file_path, 'r') as f:
        file_str = f.read()
        all_matches = get_matches_from_file(file_str)
    with open(manualy_relaunched, 'r') as f:
        all_matches_already_launched = [l.strip() for l in f.readlines()]

    matches_to_relaunch = set(all_matches).difference(set(all_matches_already_launched))
    print(len(all_matches), len(all_matches_already_launched), len(matches_to_relaunch))

    base_str = "oarsub --resubmit {}"
    with open(manualy_relaunched, 'a') as f:
        for match in matches_to_relaunch:
            sub_str = base_str.format(match)
            print(sub_str.split())
            f.write("{}\n".format(match))
            subprocess.run(sub_str.split())


if __name__ == "__main__":
    main()