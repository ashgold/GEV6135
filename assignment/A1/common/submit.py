import os
import zipfile

_A1_FILES = [
    "pytorch101.py",
    "pytorch101.ipynb",
]


def make_a1_submission(assignment_path, name=None, idnum=None, group=None):
    _make_submission(assignment_path, _A1_FILES, "A1", name, idnum, group)


def _make_submission(
    assignment_path, file_list, assignment_no, name, idnum, group
):
    if name is None or idnum is None:
        name, idnum = _get_user_info(name, idnum)
    zip_path = "{}-{}-{}.zip".format(
        name.lower().replace(' ', '_'), idnum, assignment_no
    )
    zip_path = os.path.join(assignment_path, zip_path)
    group_text = ""
    if group:
        group_text = "\n".join([f"{key} {group[key]}" for key in group])
    print("Writing zip file to: ", zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        if group_text:
            zf.writestr("group.txt", group_text)
        for filename in file_list:
            in_path = os.path.join(assignment_path, filename)
            if not os.path.isfile(in_path):
                raise ValueError('Could not find file "%s"' % filename)
            zf.write(in_path, filename)


def _get_user_info(name, idnum):
    if name is None:
        name = input("Enter your name (e.g. kibok lee): ")
    if idnum is None:
        idnum = input("Enter your id number (e.g. 2022123456): ")
    return name, idnum
