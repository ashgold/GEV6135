import os
import zipfile

FILES = {
    'A1': ['pytorch101.py', 'pytorch101.ipynb'],
    'A2': ['knn.py', 'knn.ipynb'],
}


def make_submission(
    assignment_path, anum, option=1, name=None, idnum=None, group=None
):
    if name is None or idnum is None:
        name, idnum = _get_user_info(name, idnum)
    name_str = name.lower().replace(' ', '_')
    zip_path = f'{name_str}-{idnum}-A{anum}-O{option}.zip'
    zip_path = os.path.join(assignment_path, zip_path)
    group_text = ""
    if group:
        group_text = "\n".join([f"{key} {group[key]}" for key in group])
    print('Writing zip file to: ', zip_path)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        if group_text:
            zf.writestr("group.txt", group_text)
        for filename in FILES[f'A{anum}']:
            in_path = os.path.join(assignment_path, filename)
            if not os.path.isfile(in_path):
                raise ValueError(f'Could not find file {filename}')
            zf.write(in_path, filename)


def _get_user_info(name, idnum):
    if name is None:
        name = input('Enter your name (e.g. kibok lee): ')
    if idnum is None:
        idnum = input('Enter your id number (e.g. 2022123456): ')
    return name, idnum
